# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2024, Constantino Álvarez, Tuomas Määttä, Sasan Sharifipour, Miguel Bordallo  (CMVS - University of Oulu)
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F


def chamfer_distance(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """
    Compute the Chamfer Distance between predicted and ground truth point clouds.

    Args:
        pred_points (torch.Tensor): Predicted points of shape [batch_size, num_points, 3].
        gt_points (torch.Tensor): Ground truth points of shape [batch_size, num_points, 3].

    Returns:
        torch.Tensor: Chamfer distance between the predicted and ground truth point clouds.
    """
    # Compute pairwise distances between predicted and ground truth points
    dist1 = torch.cdist(pred_points, gt_points)
    dist2 = torch.cdist(gt_points, pred_points)

    # Find nearest neighbors and compute the loss
    loss1 = dist1.min(dim=2)[0].mean(dim=1)
    loss2 = dist2.min(dim=2)[0].mean(dim=1)
    loss = loss1 + loss2

    return loss.mean()


def chamfer_loss_function(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the Chamfer Loss between predicted and target point clouds.

    Args:
        pred (torch.Tensor): Predicted points of shape [batch_size, num_points, 3].
        target (torch.Tensor): Target points of shape [batch_size, num_points, 3].

    Returns:
        torch.Tensor: Chamfer loss between the predicted and target point clouds.
    """
    # Compute pairwise distances between predicted and target points
    dist1 = torch.cdist(pred, target)

    # Compute the minimum distances for each point
    min_dist1, _ = torch.min(dist1, dim=2)  # Min distance to target for each predicted point
    min_dist2, _ = torch.min(dist1, dim=1)  # Min distance to prediction for each ground truth point

    # Return the average Chamfer loss over the batch
    return (min_dist1.mean(dim=1) + min_dist2.mean(dim=1)).mean()


# def chamfer_distance2(pred, gt):
#     """
#     Compute Chamfer Distance between two point clouds.
#
#     Args:
#         pred (torch.Tensor): Predicted point cloud, [B, N, 3]
#         gt (torch.Tensor): Ground truth point cloud, [B, M, 3]
#
#     Returns:
#         torch.Tensor: Chamfer Distance
#     """
#     # Compute pairwise distances
#     dist = torch.cdist(pred, gt, p=2)  # [B, N, M]
#
#     # For each point in pred, find the minimum distance to gt
#     min_pred_to_gt, _ = torch.min(dist, dim=2)  # [B, N]
#
#     # For each point in gt, find the minimum distance to pred
#     min_gt_to_pred, _ = torch.min(dist, dim=1)  # [B, M]
#
#     # Compute Chamfer Distance
#     loss = torch.mean(min_pred_to_gt) + torch.mean(min_gt_to_pred)
#     return loss


def feature_transform_regularizer(trans):
    """
    Regularization loss for the feature transformation matrix.

    Args:
        trans (torch.Tensor): Transformation matrix, [B, K, K]

    Returns:
        torch.Tensor: Regularization loss
    """
    K = trans.size(1)
    batchsize = trans.size(0)
    device = trans.device
    I = torch.eye(K, device=device).unsqueeze(0).repeat(batchsize, 1, 1)  # [B, K, K]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class CSI2PointCloudLoss(nn.Module):
    def __init__(self, chamfer_weight=1.0, regularization_weight=0.1):
        super(CSI2PointCloudLoss, self).__init__()
        self.chamfer_weight = chamfer_weight
        self.regularization_weight = regularization_weight

    def forward(self, predicted_points, ground_truth_points, trans_feat=None):
        # Chamfer Distance Loss
        chamfer_loss = chamfer_loss_function(predicted_points, ground_truth_points)

        if trans_feat is not None:
            # Regularization Loss
            regularization_loss = feature_transform_regularizer(trans_feat)
            # Combine Chamfer loss with regularization loss
            total_loss = self.chamfer_weight * chamfer_loss + self.regularization_weight * regularization_loss
        else:
            total_loss = chamfer_loss

        return total_loss
