import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import box_iou

def convert_center_to_cell_coords(x_center: torch.Tensor, 
                                  y_center: torch.Tensor) -> tuple[torch.Tensor, 
                                                                   torch.Tensor]:
    
    ''' Converts center coords relative to bounding box to 
       center coords relative to its grid cell. 

       Inputs: 
        x_center, y_center

       Outputs:
        c_i, c_j, x_c, y_c
    '''
    
    # finds cell index (i, j)
    c_i, c_j = (y_center * 7).to(torch.int32), (x_center * 7).to(torch.int32)

    x_c, y_c = (x_center * 7) - c_j, (y_center * 7) - c_i

    return c_i, c_j, x_c, y_c

def convert_cell_to_center_coords(c_i, c_j, x_c, y_c):

    ''' Converts center coords relative to its grid cell to 
       center coords relative to the entire image. 

       Inputs: 
        c_i, c_j, x_c, y_c, 

       Outputs:
        x_center, y_center
    '''

    x_center, y_center = (x_c + c_j) / 7, (y_c + c_i) / 7
    return x_center, y_center

def compute_minmax_from_center_and_dims(x_center, y_center, w, h):

    ''' Computes x_min, x_max, y_min, y_max of bounding box given 
        x_center, y_center, w, and h. 

       Inputs: 
        x_center, y_center, w, h 

       Outputs:
        x_min, y_min, x_max, y_max
    '''

    x = torch.concat([(x_center - (w/2)).unsqueeze(1), 
                      x_center + (w/2).unsqueeze(1)], dim = -1)
    
    y = torch.concat([y_center - (h/2).unsqueeze(1), 
                      y_center + (h/2).unsqueeze(1)], dim = -1)


    x_min, x_max = torch.min(x, dim = -1, keepdim = True)[0], torch.max(x, dim = -1, 
                                                                     keepdim = True)[0]
    
    y_min, y_max = torch.min(y, dim = -1, keepdim = True)[0], torch.max(y, dim = -1, 
                                                                     keepdim = True)[0]
    
    return torch.concat([x_min, y_min, x_max, y_max], dim = -1)

def find_responsible_box(gt_box, pred_box1, pred_box2, box1, box2):
    iou1 = torch.diag(box_iou(gt_box, pred_box1))
    iou2 = torch.diag(box_iou(gt_box, pred_box2))

    mask = iou2 > iou1 

    best_params = torch.where(mask.unsqueeze(1), box2, box1)

    return iou1, iou2, mask, best_params

def compute_no_obj_confidence(c_i, c_j, mask, preds):

    '''
    Computes the ground truth and predicted (no object) confidence scores.

    Inputs:
        c_i, c_j, mask, preds

    Outputs:
        gt_no_obj_confs, pred_no_obj_confs
    '''

    B = preds.shape[0]

    pred_confs = preds[:, :, :, 4:10:5] # (64, 7, 7, 2)
    pred_confs = pred_confs.reshape(B, 7 * 7 * 2)

    int_mask = mask.to(torch.int32)

    # compute flattened index of responsible box within predicted cell to filter out 
    best_indices = (c_i * 14 + c_j * 2 + (int_mask)).unsqueeze(1) # (64, 1)
    indices = torch.arange(pred_confs.shape[-1]).expand(pred_confs.shape[0], -1) # (64, 98)

    filter_mask = best_indices != indices # (64, 98)

    pred_no_obj_confs = pred_confs[filter_mask] # (64, 97)
    gt_no_obj_confs = torch.zeros_like(pred_no_obj_confs) # (64, 97)

    return gt_no_obj_confs, pred_no_obj_confs


class YOLOLoss(nn.Module):
    '''
    Computes YOLO loss function.

    Inputs in forward-pass:
        predictions, labels
    '''
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
    
    def forward(self, preds, labels):

        # unpack labels to store each label feature 
        class_idx, x_center, y_center, w, h = labels.transpose(0, 1)

        # one-hot encode ground truth class_idx to get class_probs
        class_probs = F.one_hot(class_idx.to(torch.long), num_classes = 200)

        # convert ground truth center coords to retrive ground truth cell indices and coords 
        c_i, c_j, x_c, y_c = convert_center_to_cell_coords(x_center, y_center)

        # find which cells were predicted/responsible using the ground truth cell indices
        pred_cells = preds[torch.arange(len(c_i)), c_i, c_j, :]

        # separate each box within the responsible cells
        box1 = pred_cells[:, :5]
        box2 = pred_cells[:, 5:10]

        # store the class probs of the responsible cells 
        class_probs_hat = pred_cells[:, 10:] 

        # unpack the features of each box 
        x_hat1, y_hat1, w_hat1, h_hat1, conf_hat1 = box1.transpose(0, 1)
        x_hat2, y_hat2, w_hat2, h_hat2, conf_hat2 = box2.transpose(0, 1)

        # compute the predicted center coords of each box given cell indices and coords
        x_center_hat1, y_center_hat1 = convert_cell_to_center_coords(c_i, c_j, x_hat1, y_hat1)
        x_center_hat2, y_center_hat2 = convert_cell_to_center_coords(c_i, c_j, x_hat2, y_hat2)

        # compute x_min, y_min, x_max, y_max of bounding box 
        gt_box = compute_minmax_from_center_and_dims(x_center, y_center, w, h)
        bbox1 = compute_minmax_from_center_and_dims(x_center_hat1, y_center_hat1, w_hat1, h_hat1)
        bbox2 = compute_minmax_from_center_and_dims(x_center_hat2, y_center_hat2, w_hat2, h_hat2)

        # compute IoUs, mask, and the params of the responsible boxes within the predicted cells
        iou1, iou2, mask, best_params = find_responsible_box(gt_box, bbox1, bbox2, box1, box2)

        # unpack the features of the params 
        x_c_hat, y_c_hat, w_hat, h_hat, conf_hat = best_params.transpose(0, 1)

        # compute ground-truth confidence (object-present) using the mask and computed IoUs
        conf = torch.where(mask, iou2, iou1)

        # compute ground-truth and predicted confidence (no object present)
        no_obj_conf, no_obj_conf_hat = compute_no_obj_confidence(c_i, c_j, mask, preds)

        # compute coord loss 
        coord_loss = self.loss_fn(x_c, x_c_hat) + self.loss_fn(y_c, y_c_hat)

        # compute dim loss
        dim_loss = (
                self.loss_fn(torch.sqrt(w), torch.sqrt(w_hat)) + 
                self.loss_fn(torch.sqrt(h), torch.sqrt(h_hat))
        )

        # apply bbox scaling factor to coord and dim loss
        bbox_loss = self.lambda_coord * (coord_loss + dim_loss)

        # compute confidence (object present) loss
        conf_loss = self.loss_fn(conf, conf_hat)

        # compute confidence (no object present) loss
        no_obj_conf_loss = self.lambda_noobj * self.loss_fn(no_obj_conf, no_obj_conf_hat)

        # compute class probs loss
        class_loss = self.loss_fn(class_probs, class_probs_hat)

        # compute YOLO loss
        return (bbox_loss + conf_loss + no_obj_conf_loss + class_loss)
    