# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2024, Constantino Álvarez, Tuomas Määttä, Sasan Sharifipour, Miguel Bordallo  (CMVS - University of Oulu)
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------


#############################################################################################################################################
#  USAGE:
#       - With datasets: python test_model.py
#  Depends on  csi2pointcloud.py and mmfi.py
#############################################################################################################################################


import os
import sys
import yaml
import torch
import torch.nn as nn
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

from PIL import Image

# Add Main_Folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mmfi.mmfi_lib.mmfi import make_dataset, make_dataloader
from src.csi2pointcloud.csi2pointcloud_arch01 import CSI2PointCloudModel
from src.loss.mmslab_loss import chamfer_distance, CSI2PointCloudLoss


###############################
#
#      HELPER FUNCTIONS
#
###############################

def figure_to_array(fig, target_size=None, dpi=400):
    """
    Convert a Matplotlib figure to a high-resolution NumPy array (RGB).
    Optionally resize to a target size (width, height) using Pillow.
    """
    fig.set_dpi(dpi)
    fig.canvas.draw()

    # Use buffer_rgba for rendering
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    img = buf.reshape(height, width, 4)  # RGBA

    # Convert RGBA to RGB using Pillow
    img = Image.fromarray(img).convert("RGB")

    # Resize to target size if specified
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)  # Updated method for resizing

    return np.array(img)


def visualize_point_clouds(gt_lidar_frame, pred_lidar_frame, frame_index, subject="S01", scene="E01", action="A01"):
    # Remove zero points from both ground truth and predicted point clouds
    gt_mask = np.any(gt_lidar_frame != 0, axis=1)
    pred_mask = np.any(pred_lidar_frame != 0, axis=1)

    gt_points = gt_lidar_frame[gt_mask]
    pred_points = pred_lidar_frame[pred_mask]

    # Compute distances for color mapping
    gt_distances = np.linalg.norm(gt_points, axis=1)
    pred_distances = np.linalg.norm(pred_points, axis=1)

    # Create a 3D scatter plot for both ground truth and predicted point clouds
    fig = plt.figure(figsize=(19.2, 10.1))

    # Global title
    plt.suptitle(f'Frame {frame_index} | Subject: {subject} | Scene: {scene} | Action: {action}', fontsize=20)

    # Ground truth point cloud (left subplot)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim([1.0, 2.0])
    ax1.set_ylim([-2.0, 1.2])
    ax1.set_zlim([-1.2, 0.1])
    ax1.set_autoscale_on(False)
    sc1 = ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                      c=gt_distances, cmap='viridis', s=7, alpha=0.9)
    ax1.set_title(f'Ground Truth Point Cloud Frame {frame_index}', fontsize=18)
    ax1.set_xlabel('Z (meters)', fontsize=14)
    ax1.set_ylabel('X (meters)', fontsize=14)
    ax1.set_zlabel('Y (meters)', fontsize=14)
    ax1.view_init(elev=20, azim=165)
    # ax1.set_box_aspect([np.ptp(a) for a in [gt_points[:, 0], gt_points[:, 1], gt_points[:, 2]]])
    ax1.set_box_aspect([1.16, 3.0, 1.1])
    # for a in [gt_points[:, 0], gt_points[:, 1], gt_points[:, 2]]:
    #     print(np.ptp(a))

    # Add color bar for ground truth
    # cbar1 = fig.colorbar(sc1, ax=ax1, label='Distance from Sensor (meters)', pad=0.1, shrink=0.5, aspect=10)
    # cbar1.ax.tick_params(labelsize=14)

    cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.1, shrink=0.5)  # fraction and pad for size
    cbar1.set_label('Distance from Sensor (meters)', fontsize=14)  # set the label and the size
    cbar1.ax.tick_params(labelsize=10)  # size of the ticks

    # Predicted point cloud (right subplot)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim([1.0, 2.0])
    ax2.set_ylim([-2.0, 1.2])
    ax2.set_zlim([-1.2, 0.1])
    ax2.set_autoscale_on(False)
    sc2 = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                      c=pred_distances, cmap='viridis', s=7, alpha=0.9)
    ax2.set_title(f'Predicted Point Cloud Frame {frame_index}', fontsize=18)
    ax2.set_xlabel('Z (meters)', fontsize=14)
    ax2.set_ylabel('X (meters)', fontsize=14)
    ax2.set_zlabel('Y (meters)', fontsize=14)
    ax2.view_init(elev=20, azim=165)
    ax2.set_box_aspect([1.16, 3.0, 1.1])
    # ax2.set_box_aspect([np.ptp(a) for a in [pred_points[:, 0], pred_points[:, 1], pred_points[:, 2]]])

    # Add color bar for predicted point cloud
    # cbar2 = fig.colorbar(sc2, ax=ax2, label='Distance from Sensor (meters)', pad=0.1, shrink=0.5, aspect=10)
    # cbar2.ax.tick_params(labelsize=14)

    cbar1 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.1, shrink=0.5)  # fraction and pad for size
    cbar1.set_label('Distance from Sensor (meters)', fontsize=14)  # set the label and the size
    cbar1.ax.tick_params(labelsize=10)  # size of the ticks

    plt.tight_layout()
    # img_array = figure_to_array(fig, target_size=(1920, 1080))  # Force resize

    plt.show()
    # plt.close(fig)
    # return img_array
    return 0


def save_video(output_images, output_path, frame_rate=10):
    height, width, _ = output_images[0].shape
    vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))
    for img in output_images:
        vid_writer.write(img)
    vid_writer.release()


########################################
#
#      METRICS
#
#########################################


import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff


def icp_registration(pc1, pc2, threshold=0.02):
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pc1, pc2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return reg_p2p.fitness, reg_p2p.inlier_rmse


def icp_registration2(pc1, pc2, threshold=0.02):
    # Check if point clouds are empty
    if len(pc1.points) == 0 or len(pc2.points) == 0:
        raise ValueError("One or both point clouds are empty.")

    # Initial transformation
    trans_init = np.identity(4)

    # Perform ICP registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pc1, pc2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return reg_p2p.fitness, reg_p2p.inlier_rmse


# Validate point clouds (ensure no NaNs or infinities)
def validate_point_cloud(pcd):
    points = np.asarray(pcd.points)
    if not np.all(np.isfinite(points)):
        raise ValueError("Point cloud contains NaNs or infinite values.")


def get_icp_values(gt_lidar_frame, pred_lidar_frame):
    # # Convert NumPy arrays to Open3D PointCloud
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(gt_lidar_frame)
    pcd2.points = o3d.utility.Vector3dVector(pred_lidar_frame)

    # Compute ICP registration
    fitness, rmse = icp_registration2(pcd1, pcd2)
    return fitness, rmse

###############################
#
#      TEST FUNCTIONS
#
###############################


def test_epoch(model, device, dataloader, loss_fn, dataset_type="Validation", visualize=True):
    model.eval()
    val_loss = []

    average_icp = []
    average_icprmse = []
    index_total_time = 0
    total_time = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):

            t0 = time.time()

            wifi_csi_frame = batch['input_wifi-csi'].to(device)
            gt_point_cloud = batch['input_lidar'].to(device)

            wifi_csi_frame = (wifi_csi_frame - wifi_csi_frame.mean()) / wifi_csi_frame.std()
            gt_point_cloud = (gt_point_cloud - gt_point_cloud.mean()) / gt_point_cloud.std()

            # Forward pass
            pred_point_cloud, trans_feat = model(wifi_csi_frame)
            total_time += time.time() - t0
            index_total_time += len(batch['input_lidar'])

            # Compute loss
            # loss = loss_fn(pred_point_cloud, gt_point_cloud, trans_feat)
            loss = loss_fn(pred_point_cloud, gt_point_cloud)
            val_loss.append(loss.item())

            # Metric comparison
            for frame_idx in range(len(batch['input_lidar'])):
                gt_lidar_frame = gt_point_cloud[frame_idx].cpu().numpy()  # First frame of the batch
                pred_lidar_frame = pred_point_cloud[frame_idx].cpu().numpy()  # First predicted frame of the batch
                fitness, rmse = get_icp_values(gt_lidar_frame, pred_lidar_frame)
                # print(f"    >> [CSI2PC Test Results] ICP Fitness f:{frame_idx}/b:{batch_idx}: {fitness}, ICP RMSE: {rmse}")
                average_icp.append(fitness)
                average_icprmse.append(rmse)

            # Visualize results for the first batch (or modify as needed)
            if visualize:
                for frame_idx in range(len(batch['input_lidar'])):
                    print(f"    >> [CSI2PC Test Results] Plotting: Frame {frame_idx} int batch {batch_idx}/{len(dataloader)}")

                    gt_lidar_frame = gt_point_cloud[frame_idx].cpu().numpy()  # First frame of the batch
                    pred_lidar_frame = pred_point_cloud[frame_idx].cpu().numpy()  # First predicted frame of the batch

                    # # Convert NumPy arrays to Open3D PointCloud
                    pcd1 = o3d.geometry.PointCloud()
                    pcd2 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector(gt_lidar_frame)
                    pcd2.points = o3d.utility.Vector3dVector(pred_lidar_frame)

                    output_image = visualize_point_clouds(gt_lidar_frame, pred_lidar_frame, frame_index=batch['idx'][frame_idx],
                                                          subject=batch['subject'][frame_idx], scene=batch['scene'][frame_idx], action=batch['action'][frame_idx])

                    # output_images.append(output_image)

        avg_loss = np.mean(val_loss)
        print(f"    >> [CSI2PC Test Results] {dataset_type} loss: {avg_loss:.6f}")

        print(f"    >> [CSI2PC Test Results] Average ICP loss: {np.mean(average_icp):.6f}")
        print(f"    >> [CSI2PC Test Results] Average ICP RMSE loss: {np.mean(average_icprmse):.6f}")
        print(f"    >> [CSI2PC Test Results] Total time: {total_time:.2f}")
        print(f"    >> [CSI2PC Test Results] Total images testing: {index_total_time}")
        print(f"    >> [CSI2PC Test Results] Average time per estimation: {(total_time / index_total_time) * 1000:.4f}")

        # save_video(output_images, "output_video_scsi2pc_arch01_v4_S02_A16.mp4")

        return avg_loss


###############################
#
#      MAIN FUNCTION
#
###############################
def main():
    """
    This script tests a trained CSI2PointCloud model to estimate 3D Point Clouds using WiFi CSI data.

    **Usage:**

    - Select a trained model file or checkpoint.
    - Select a user or users in the MM-Fi database to visualize the inference.
    - Evaluate the model according to the metrics and protocol described in the original publication.

    """
    print('\n')
    print('*******************************************************************************')
    print('*          MMSLAB CSI2PointCloud: Starting to test model Arch01_v1           *')
    print('*******************************************************************************')
    print('\n')

    print(f"[INFO]")
    print(f"  * Model name: Arch01")
    print(f"  * Model type: Encoder - Decoder")
    print(f"  * Model encoder: temporal encoder and transformers")
    print(f"  * Model decoder: Linear Projection")

    #############################
    #       LOAD DATASET        #
    #############################
    yaml_config_file = "../data/configurations/config_mmfi_csi2pointcloud.yaml"
    # dataset_root = "/media/arritmic/MMST003/DATABASES/Joint_Comm_and_Sensing/MMFI-Dataset/data3"
    dataset_root = "/path/to/the/MMFI-Dataset/data"
    print(f"  * Config file: {yaml_config_file}")
    print(f"  * Dataset path: {dataset_root}")
    print('\n')

    with open(yaml_config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config, subsampling=10, fixed_lidar_points=1200, frame_limit=200)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    print(f"[TESTING]")
    print(f"    >> [CSI2PC Test] Testset samples: {len(val_loader)}")
    print(f"    >> [CSI2PC Test] Selected data split: {config['split_to_use']}")

    #############################
    #        MODEL CONFIG       #
    #############################
    torch.cuda.empty_cache()

    # Initialize model, optimizer, loss function, and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = CSI2PointCloudModel(
        embedding_dim=512,
        num_heads=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_points=1200,  # Updated to match the fixed number of points
        num_antennas=3,
        num_subcarriers=114,
        num_time_slices=10
    ).to(device)

    # Download model from: https://drive.google.com/drive/folders/1U9hMGtMoQWgP_Obi5k1Vts_WbEHi85Vi?usp=sharing
    model_path = "../models/checkpoints/checkpoint_epoch_41_Arch01_v1.pth"
    print(f"    >> [CSI2PC Test] Loading checkpoint: {model_path} .....")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    loss_fn = chamfer_distance

    # loss_fn = CSI2PointCloudLoss(chamfer_weight=0.85, regularization_weight=0.15)
    t0 = time.time()

    #############################
    #        MODEL TESTING      #
    #############################
    # Validation
    val_loss = test_epoch(
        model=model,
        device=device,
        dataloader=val_loader,
        loss_fn=loss_fn,
        dataset_type="Validation",
        visualize=True
    )

    print(f"    >> [CSI2PC Test] Done! Total testing time: {(time.time() - t0) / 60:.2f} minutes")


if __name__ == '__main__':
    main()
