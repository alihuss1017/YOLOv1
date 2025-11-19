import torch 
import torch.nn as nn
from torchvision.ops.boxes import box_iou
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.loss_fn = nn.MSELoss()

    def _get_predicted_grid_cell(self, labels: torch.Tensor):

        '''
        Returns predicted cell (i, j).
        input: torch.Tensor of shape (64, )
        outputs: torch.Tensor, torch.Tensor of shape (64, )
        '''

        x_center, y_center = labels[:, 1], labels[:, 2]

        # cell_i is row-index, cell_j is col-index.
        c_i, c_j = (y_center / (1/7)).to(torch.int32), (x_center / (1/7)).to(torch.int32)

        return c_i, c_j
    
    def _get_predicted_box_params(self, predicted_cells: torch.Tensor):

        '''
        Returns the responsible box's parameters (x, y, w, h, confidence). 
        Also returns the predicted grid cell's class probabilities.
        '''

        # conf1, conf2 represent the confidence scores of bounding box 1 and 2 respectively. 
        conf1 = predicted_cells[:, 4]
        conf2 = predicted_cells[:, 9]

        # this is our predicted class-conditional probabilities
        probs_hat = predicted_cells[:, 10: ]

        # this is our mask; True/1 if box 1's confidence is greater than box 2's, else False/0.
        conf_mask = conf2 > conf1 

        
        # we apply our mask; if true, then we take predicted_cells[:, 5:10] else 
        # predicted_cells[:, 0:5]. the shape will be (64, 5)
        selected_box_params = torch.where(conf_mask.unsqueeze(1), predicted_cells[:, 5:10], predicted_cells[:, 0:5],)

        # now we can get our predicted values for loss function. each var is of shape (64, )
        x_hat_c, y_hat_c, w_hat, h_hat, conf_hat = selected_box_params.transpose(0, 1)
        
        return x_hat_c, y_hat_c, w_hat, h_hat, conf_hat, probs_hat

    def _get_cell_coords(self, labels: torch.Tensor, c_j: torch.Tensor, 
                         c_i: torch.Tensor):

        '''
        Returns ground truth coords of center of object relative to grid cell.
        input: torch.Tensor of shape (64, )
        outputs: torch.Tensor, torch.Tensor of shape (64, )
        '''

        x_center, y_center = labels[:, 1], labels[:, 2]

        x_c = x_center / (1/7) - c_j # (64, )
        y_c = y_center / (1/7) - c_i # (64, )

        return x_c, y_c
    
    def _get_bounding_box(self, x_center: torch.Tensor, y_center: torch.Tensor, 
                          w: torch.Tensor, h: torch.Tensor):
        '''Computes and returns the bounding box of shape (64, 4)
        given the center coords, width, and height.'''

        # to compute our ground truth bounding box, we need to find x_min, y_min, x_max, y_max
        x_min, x_max = x_center - (w/2), x_center + (w/2)
        y_min, y_max = y_center - (h/2), y_center + (h/2)

        # we unsqueeze our tensor to be of shape (64, 1) to later concatenate
        x_min, x_max = x_min.unsqueeze(1), x_max.unsqueeze(1)
        y_min, y_max = y_min.unsqueeze(1), y_max.unsqueeze(1)

        # we concatenate along columns to get tensor of shape (64, 4)
        bounding_box = torch.concat([x_min, y_min, x_max, y_max], dim = -1)

        return bounding_box

    def _get_noobj_pred_confidence(self, box_params: torch.Tensor,
                               predicted_cells: torch.Tensor, c_i: torch.Tensor,
                               c_j: torch.Tensor, batch_size: int):

        '''
        Returns the no-object predicted confidence scores, filtering out the scores 
        of the predicted grid cells. 
        '''
        box_params = box_params.view(batch_size, 7, 7, 2, 5)
        box_params = box_params.contiguous().view(batch_size, 98, 5)

        # (batch_size, 98)
        confidence_scores = box_params[..., 4]

        # conf1, conf2 represent the confidence scores of bounding box 1 and 2 respectively. 
        conf1 = predicted_cells[:, 4]
        conf2 = predicted_cells[:, 9]

        # this is our mask; True/1 if box 1's confidence is greater than box 2's, else False/0.
        conf_mask = conf2 > conf1 

        # cast as integer for later mathematical operation
        conf_int_mask = conf_mask.to(torch.int32)

        # this (batch_size, ) shaped tensor contains all indices to filter out. 
        no_obj_mask = (c_i * 14 + c_j * 2 + (conf_int_mask))

        # (98, ) shaped tensor containing all flattened indices for masking purposes
        indices = torch.arange(confidence_scores.shape[1]).unsqueeze(0).expand(batch_size, -1)

        # define keep mask to determine which values to keep
        keep_mask = (indices != no_obj_mask[:, None])

        # apply mask
        noobj_confidence_hat = confidence_scores[keep_mask].view(batch_size, 97)

        return noobj_confidence_hat

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor):
        '''
        predictions: (batch_size, 7, 7, 210)
        labels: (batch_size, 5)
        '''

        batch_size = labels.shape[0]

        # extract ground truths
        class_idx, x_center, y_center, w, h = labels[:, ].transpose(0, 1)

        # generate one-hot encoded gt class probs (64, 200)
        encoded_class_labels = F.one_hot(class_idx, num_classes = 200)

        # retrieve grid cell indices
        c_i, c_j = self._get_predicted_grid_cell(labels)

        # retrieve grid cell coords
        x_c, y_c = self._get_cell_coords(labels, c_j, c_i)

        # this will give us the predicted cell from each batch item. the predicted cell is (210, )
        predicted_cells = predictions[torch.arange(batch_size), c_i, c_j, :] # (64, 210)

        # all predicted params
        x_hat_c, y_hat_c, w_hat, h_hat, conf_hat, probs_hat = self._get_predicted_box_params(predicted_cells)

        # ground truth bounding box
        gt_bound_box = self._get_bounding_box(x_center, y_center, w, h)

        # to compute our predictions bounding box, we first need to find x_hat_center, y_hat_center. 
        x_hat_center, y_hat_center = (x_hat_c + c_j) / 7, (y_hat_c + c_i) / 7
        pred_bound_box = self._get_bounding_box(x_hat_center, y_hat_center, w_hat, h_hat)

        # computing ground truth confidence values
        gt_confidence = torch.diag(box_iou(gt_bound_box, pred_bound_box))

        box_params = predictions[..., :10]

        # generate predicted noobj confidence tensor
        noobj_confidence_hat = self._get_noobj_pred_confidence(box_params,
                                                               predicted_cells, c_i, c_j,
                                                               batch_size)
        
        # generate ground-truth noobj confidence tensor
        noobj_confidence = torch.zeros_like(noobj_confidence_hat)

        # compute first term in loss equation (x, y)
        coord_loss = self.loss_fn(x_c, x_hat_c) + self.loss_fn(y_c, y_hat_c)

        # compute second term in loss equation (w, h)
        area_loss = self.loss_fn(w, w_hat) + self.loss_fn(h, h_hat)

        # sum first and second term together with scaling factor lambda_coord
        positional_loss = self.lambda_coord * (coord_loss + area_loss)

        # compute confidence loss (object)
        obj_conf_loss = self.loss_fn(gt_confidence, conf_hat)

        # compute scaled confidence loss (no object)
        noobj_conf_loss = self.lambda_noobj * self.loss_fn(noobj_confidence, noobj_confidence_hat)

        # compute sum of object and no object confidence loss
        conf_loss = obj_conf_loss + noobj_conf_loss

        # compute class-conditional loss
        class_prob_loss = nn.MSELoss(encoded_class_labels, probs_hat)

        # return total loss
        return positional_loss + conf_loss + class_prob_loss