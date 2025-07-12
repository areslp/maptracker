import torch

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners import AssignResult
from mmdet.models.task_modules.assigners import BaseAssigner
from mmdet.models.task_modules import build_match_cost
from scipy.optimize import linear_sum_assignment

import numpy as np

@TASK_UTILS.register_module()
class HungarianLinesAssigner(BaseAssigner):
    """
        Computes one-to-one matching between predictions and ground truth.
        This class computes an assignment between the targets and the predictions
        based on the costs. The costs are weighted sum of three components:
        classification cost and regression L1 cost. The
        targets don't include the no_object, so generally there are more
        predictions than targets. After the one-to-one matching, the un-matched
        are treated as backgrounds. Thus each query prediction will be assigned
        with `0` or a positive integer indicating the ground truth index:
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt
        Args:
            cls_weight (int | float, optional): The scale factor for classification
                cost. Default 1.0.
            bbox_weight (int | float, optional): The scale factor for regression
                L1 cost. Default 1.0.
    """

    def __init__(self,
                 cost=dict(
                     type='MapQueriesCost',
                     cls_cost=dict(type='ClassificationCost', weight=1.),
                     reg_cost=dict(type='LinesCost', weight=1.0),
                    ),
                 **kwargs):

        self.cost = build_match_cost(cost)

    def assign(self,
               preds: dict,
               gts: dict,
               track_info=None,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """
            Computes one-to-one matching based on the weighted costs.
            This method assign each query prediction to a ground truth or
            background. The `assigned_gt_inds` with -1 means don't care,
            0 means negative sample, and positive number is the index (1-based)
            of assigned gt.
            The assignment is done in the following steps, the order matters.
            1. assign every prediction to -1
            2. compute the weighted costs
            3. do Hungarian matching on CPU based on the costs
            4. assign all to 0 (background) first, then for each matched pair
            between predictions and gts, treat this prediction as foreground
            and assign the corresponding gt index (plus 1) to it.
            Args:
                lines_pred (Tensor): predicted normalized lines:
                    [num_query, num_points, 2]
                cls_pred (Tensor): Predicted classification logits, shape
                    [num_query, num_class].

                lines_gt (Tensor): Ground truth lines
                    [num_gt, num_points, 2].
                labels_gt (Tensor): Label of `gt_bboxes`, shape (num_gt,).
                gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                    labelled as `ignored`. Default None.
                eps (int | float, optional): A value added to the denominator for
                    numerical stability. Default 1e-7.
            Returns:
                :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        
        num_gts, num_lines = gts['lines'].size(0), preds['lines'].size(0)

        # 1. assign -1 by default
        assigned_gt_inds = \
            preds['lines'].new_full((num_lines,), -1, dtype=torch.long)
        assigned_labels = \
            preds['lines'].new_full((num_lines,), -1, dtype=torch.long)

        if num_gts == 0 or num_lines == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels), None

        # 2. compute the weighted costs
        gt_permute_idx = None # (num_preds, num_gts)
        if self.cost.reg_cost.permute:
            cost, gt_permute_idx, reg_cost = self.cost(preds, gts)
        else:
            cost, reg_cost = self.cost(preds, gts)
        
        # Manipulate the cost matrix here using the two-frame matching info
        # for non-first-frame supervision
        if track_info is not None:
            prop_i = 0
            # iterate through queries
            for j in range(cost.shape[0]):
                if j >= len(track_info['track_queries_fal_pos_mask']):
                    # padding queries, loss will be filtered later
                    cost[j] = np.inf
                    continue
                    
                if track_info['track_queries_fal_pos_mask'][j]:
                    # false positive and palceholder track queries should not
                    # be matched to any target
                    cost[j] = np.inf
                
                # Tweak the cost matrix here to force the G.T. assignment of the track queries
                elif track_info['track_queries_mask'][j]:
                    track_query_id = track_info['track_query_match_ids'][prop_i].long().item()
                    prop_i += 1

                    cost[j] = np.inf
                    cost[:, track_query_id] = np.inf
                    cost[j, track_query_id] = -1

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu().numpy()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        try:
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        except:
            print('cost max{}, min{}'.format(cost.max(), cost.min()))
            import pdb; pdb.set_trace()

        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            preds['lines'].device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            preds['lines'].device)
        
        # Pass out the un-weighted reg cost for temporal propagation
        mathced_reg_cost = reg_cost[matched_row_inds, matched_col_inds]        

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gts['labels'][matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels), gt_permute_idx, mathced_reg_cost