# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2024, Constantino Álvarez, Tuomas Määttä, Sasan Sharifipour, Miguel Bordallo  (CMVS - University of Oulu)
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------


#############################################################################################################################################
#  USAGE:
#       - With datasets: python train_model.py
#  Depends on  csi2pointcloud.py and mmfi.py
#############################################################################################################################################

import os
import sys
import yaml
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
import random
import copy
from torch.amp import autocast, GradScaler
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F

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
    fig = plt.figure(figsize=(18, 8))

    # Global title
    plt.suptitle(f'Frame {frame_index + 1} | Subject: {subject} | Scene: {scene} | Action: {action}', fontsize=16)

    # Ground truth point cloud (left subplot)
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                      c=gt_distances, cmap='viridis', s=5, alpha=0.7)
    ax1.set_title(f'Ground Truth Point Cloud Frame {frame_index + 1}')
    ax1.set_xlabel('Z (meters)')
    ax1.set_ylabel('X (meters)')
    ax1.set_zlabel('Y (meters)')
    ax1.set_box_aspect([np.ptp(a) for a in [gt_points[:, 0], gt_points[:, 1], gt_points[:, 2]]])

    # Add color bar for ground truth
    cbar1 = fig.colorbar(sc1, ax=ax1, label='Distance from Sensor (meters)', pad=0.1)

    # Predicted point cloud (right subplot)
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                      c=pred_distances, cmap='plasma', s=5, alpha=0.7)
    ax2.set_title(f'Predicted Point Cloud Frame {frame_index + 1}')
    ax2.set_xlabel('Z (meters)')
    ax2.set_ylabel('X (meters)')
    ax2.set_zlabel('Y (meters)')
    ax2.set_box_aspect([np.ptp(a) for a in [pred_points[:, 0], pred_points[:, 1], pred_points[:, 2]]])

    # Add color bar for predicted point cloud
    cbar2 = fig.colorbar(sc2, ax=ax2, label='Distance from Sensor (meters)', pad=0.1)

    plt.tight_layout()
    plt.show()


def normalize_point_cloud(pc):
    """
    Normalizes a point cloud to have zero mean and unit variance.

    Args:
        pc (torch.Tensor): Point cloud of shape [B, N, 3].

    Returns:
        torch.Tensor: Normalized point cloud.
    """
    centroid = pc.mean(dim=1, keepdim=True)
    pc = pc - centroid
    furthest_distance = pc.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    pc = pc / furthest_distance
    return pc


def collate_fn_padd_csi2pc(batch):
    '''
    Pads batch of variable length if necessary and collates the batch.
    Assumes that all 'wifi-csi' samples have consistent shapes [A, S, 2, T].
    '''

    batch_data = {
        'modality': batch[0]['modality'],
        'scene': [sample['scene'] for sample in batch],
        'subject': [sample['subject'] for sample in batch],
        'action': [sample['action'] for sample in batch],
        'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None
    }

    # Collate 'output'
    _output = [sample['output'] for sample in batch]  # Each is [Npoints, 3]
    _output = torch.stack(_output, dim=0)  # Shape: [B, Npoints, 3]
    batch_data['output'] = _output

    for mod in batch_data['modality']:
        if mod in ['mmwave', 'lidar']:
            # Assuming 'mmwave' and 'lidar' have their own handling
            _input = [torch.Tensor(sample['input_' + mod]) for sample in batch]
            _input = torch.nn.utils.rnn.pad_sequence(_input, batch_first=True)
            batch_data['input_' + mod] = _input
        else:
            # Assuming 'wifi-csi' has shape [A, S, 2, T]
            _input = [torch.FloatTensor(sample['input_' + mod]) for sample in batch]
            _input = torch.stack(_input, dim=0)  # Shape: [B, A, S, 2, T]
            batch_data['input_' + mod] = _input

    return batch_data


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, name_model="default"):
    """
    Save a checkpoint of the model state, optimizer state, epoch, and loss.

    :param model: Model to save.
    :param optimizer: Optimizer to save.
    :param epoch: Current epoch.
    :param loss: Loss value at this epoch.
    :param checkpoint_dir: Directory to save checkpoints.
    """
    checkpoint_path = f"{checkpoint_dir}checkpoint_epoch_{epoch + 1}_{name_model}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'embeddings': model.embedding_dim,
        'head_num': model.num_heads,
        'antennas_num': model.num_antennas,
        'subcarriers_num': model.num_subcarriers,
        'cloud_points_num': model.num_points,
    }, checkpoint_path)
    print(f"     >> Checkpoint saved at: {checkpoint_path}")


########################################
#
#      DATA AUGMENTATION FUNCTIONS
#
#########################################

def add_gaussian_noise(csi_data, magnitude_std=0.01, phase_std=0.01):
    # Separate magnitude and phase
    magnitude = csi_data[:, :, 0, :]  # Shape: [A, S, T]
    phase = csi_data[:, :, 1, :]  # Shape: [A, S, T]

    # Add noise
    magnitude_noise = magnitude + torch.randn_like(magnitude) * magnitude_std
    phase_noise = phase + torch.randn_like(phase) * phase_std

    # Combine back
    csi_noisy = torch.stack((magnitude_noise, phase_noise), dim=2)  # Correct stacking
    return csi_noisy  # Shape: [A, S, 2, T]


def apply_phase_shift(csi_data, max_shift=np.pi):
    # Generate random phase shifts
    phase_shift = torch.rand(1) * 2 * max_shift - max_shift  # Scalar shift
    csi_data_shifted = csi_data.clone()
    csi_data_shifted[:, :, 1, :] += phase_shift  # Shift phase component
    return csi_data_shifted


def scale_magnitude(csi_data, scale_range=(0.9, 1.1)):
    # Generate random scaling factor
    scale = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
    csi_data_scaled = csi_data.clone()
    csi_data_scaled[:, :, 0, :] *= scale  # Scale magnitude component
    return csi_data_scaled


def shift_time(csi_data, max_shift=2):
    shift = np.random.randint(-max_shift, max_shift + 1)
    T = csi_data.shape[3]  # Correct index for time dimension
    shift = shift % T  # Ensure shift is within valid range
    csi_data_shifted = torch.roll(csi_data, shifts=shift, dims=3)  # Correct dim
    return csi_data_shifted  # Shape: [A, S, 2, T]


def simulate_multipath(csi_data, num_paths=2, delay_max=5, attenuation_range=(0.5, 1.0)):
    csi_augmented = csi_data.clone()
    T = csi_data.shape[3]  # Time dimension index

    for _ in range(num_paths):
        # Generate random delay and attenuation
        delay = torch.randint(1, min(delay_max + 1, T), (1,)).item()
        attenuation = torch.rand(1).item() * (attenuation_range[1] - attenuation_range[0]) + attenuation_range[0]

        # Create delayed and attenuated version
        delayed_csi = csi_data[:, :, :, :-delay]  # Shape: [A, S, 2, T - delay]

        # Pad the time dimension at the beginning to maintain shape
        pad_size = delay
        padding = torch.zeros((csi_augmented.shape[0], csi_augmented.shape[1], csi_augmented.shape[2], pad_size), dtype=csi_augmented.dtype, device=csi_augmented.device)
        delayed_csi_padded = torch.cat((padding, delayed_csi), dim=3)  # Shape: [A, S, 2, T]

        # Apply attenuation
        delayed_csi_padded *= attenuation

        # Combine with original CSI
        csi_augmented += delayed_csi_padded

    return csi_augmented  # Shape: [A, S, 2, T]


def augment_csi_data(csi_data, number=0):
    num_augm = random.randint(0, 4)
    # Apply one augmentation based on the number
    if num_augm == 0:
        csi_data = add_gaussian_noise(csi_data)
    elif num_augm == 1:
        csi_data = apply_phase_shift(csi_data)
    elif num_augm == 2:
        csi_data = scale_magnitude(csi_data)
    elif num_augm == 3:
        csi_data = shift_time(csi_data)
    elif num_augm == 4:
        csi_data = simulate_multipath(csi_data)
    return csi_data


def create_augmented_dataset(csi_dataset, num_augmentations=3):
    augmented_data = []

    index = 0
    amount_data = len(csi_dataset)
    for sample in csi_dataset:
        augmented_data.append(sample)  # Add the original sample

        csi_sample = sample['input_wifi-csi']
        gt_point_cloud = sample['input_lidar']

        # Apply augmentations and add to the dataset
        for idx in range(num_augmentations):
            csi_augmented = augment_csi_data(csi_sample)
            sample_cop = copy.deepcopy(sample)
            sample_cop['input_wifi-csi'] = csi_augmented
            # 'input_lidar' remains the same
            augmented_data.append(sample_cop)
        if index % 100 == 0:
            print(f"  >> Data augmentation. Sample {index}/{amount_data}")
        index += 1

    return augmented_data


class AugmentedMMFiDataset(torch.utils.data.Dataset):
    def __init__(self, augmented_data):
        self.data = augmented_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


########################################
#
#      TRAINING AND TESTING FUNCTIONS
#
########################################
def print_model_settings(model):
    """
    Prints a summary of the model settings.

    Args:
        model: The model whose settings are to be printed.
    """
    # Extract settings from the model
    settings = {
        "Embedding Dimension": model.embedding_dim,
        "Number of Points": model.num_points,
        "Number of Antennas": model.num_antennas,
        "Number of Subcarriers": model.num_subcarriers,
        "Number of Time Slices": model.num_time_slices,
        "Number of Heads": model.num_heads,
    }

    # Print settings in a formatted way
    print("[Model Settings Summary]")
    print("-" * 40)
    for key, value in settings.items():
        print(f"  * {key:<30}: {value}")
    print("-" * 40)


def train_epoch(model, device, dataloader, loss_fn, optimizer, epoch):
    model.train()
    train_loss = []

    # Get the total number of batches for progress tracking
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # Extract inputs and outputs from the batch
        wifi_csi_frame = batch['input_wifi-csi'].to(device)  # Input data
        gt_point_cloud = batch['input_lidar'].to(device)

        # Data normalization
        wifi_csi_frame = (wifi_csi_frame - wifi_csi_frame.mean()) / wifi_csi_frame.std()
        gt_point_cloud = (gt_point_cloud - gt_point_cloud.mean()) / gt_point_cloud.std()
        # gt_point_cloud = normalize_point_cloud(gt_point_cloud)

        # Zero the parameter gradients
        optimizer.zero_grad()

        pred_point_cloud, trans_feat = model(wifi_csi_frame)  # Shape: [batch_size, num_points, 3]
        loss = loss_fn(pred_point_cloud, gt_point_cloud, trans_feat)

        loss.backward()
        optimizer.step()

        # Record loss
        train_loss.append(loss.item())

        # Print the progress for each batch
        if (batch_idx + 1) % 20 == 0:
            print(f"     - Epoch [{epoch + 1}], Batch [{batch_idx + 1}/{total_batches}], Loss: {loss.item():.6f}")

    # Compute and print the average loss for the epoch
    avg_loss = np.mean(train_loss)
    print(f"     - Epoch [{epoch + 1}] >> Average Training Loss: {avg_loss:.6f}")

    return avg_loss


def test_epoch(model, device, dataloader, loss_fn, epoch, dataset_type="Validation", visualize=True):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and outputs from the batch
            wifi_csi_frame = batch['input_wifi-csi'].to(device)
            gt_point_cloud = batch['input_lidar'].to(device)

            wifi_csi_frame = (wifi_csi_frame - wifi_csi_frame.mean()) / wifi_csi_frame.std()
            gt_point_cloud = (gt_point_cloud - gt_point_cloud.mean()) / gt_point_cloud.std()
            # gt_point_cloud = normalize_point_cloud(gt_point_cloud)

            # Forward pass
            pred_point_cloud, trans_feat = model(wifi_csi_frame)

            # Compute loss
            loss = loss_fn(pred_point_cloud, gt_point_cloud, trans_feat)
            val_loss.append(loss.item())

            # Visualize results for the first batch (or modify as needed)
            if visualize and epoch % 5 == 0:
                for frame_idx in range(wifi_csi_frame.size(0)):
                    if frame_idx % 10 == 0:
                        subject = batch['subject'][frame_idx]
                        scene = batch['scene'][frame_idx]
                        action = batch['action'][frame_idx]
                        print(f"     >> Saving: Frame {frame_idx} in batch {batch_idx}/{len(dataloader)}. Other info: {subject} {scene} {action}")
                        gt_lidar_frame = gt_point_cloud[frame_idx].cpu().numpy()  # First frame of the batch
                        pred_lidar_frame = pred_point_cloud[frame_idx].cpu().numpy()  # First predicted frame of the batch

                        visualize_point_clouds(gt_lidar_frame, pred_lidar_frame, frame_index=frame_idx,
                                               subject=batch['subject'][frame_idx], scene=batch['scene'][frame_idx], action=batch['action'][frame_idx])

    avg_loss = np.mean(val_loss)
    print(f"     >> {dataset_type} loss: {avg_loss:.6f}")
    return avg_loss


###############################
#
#      MAIN FUNCTION
#
###############################
def main():
    print('\n')
    print('*******************************************************************************')
    print('*          MMSLAB CSI2PointCloud: Starting to train model Arch01_v1           *')
    print('*******************************************************************************')
    print('\n')

    print(f"[INFO]")
    print(f"  * Model name: Arch01")
    print(f"  * Model type: Encoder - Decoder")
    print(f"  * Model encoder: temporal encoder and transformers")
    print(f"  * Model decoder: Linear Projection")
    date_train = datetime.datetime.now().strftime('%d-%b-%Y_(%H:%M:%S)')
    print(f"  * Date: {date_train}")

    #############################
    #       LOAD DATASET        #
    #############################
    yaml_config_file = "../data/configurations/config_mmfi_csi2pointcloud.yaml"
    # dataset_root = "/media/arritmic/MMST003/DATABASES/Joint_Comm_and_Sensing/MMFI-Dataset/data"
    dataset_root = "/path/to/the/MMFI-Dataset/data"
    print(f"  * Config file: {yaml_config_file}")
    print(f"  * Dataset path: {dataset_root}")

    with open(yaml_config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config, subsampling=5, fixed_lidar_points=1200)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    print(f"[TRAINING]")
    print(f"    >> [CSI2PC Train] Trainset samples: {len(train_loader)}. Batch size: {config['train_loader']['batch_size']}")
    print(f"    >> [CSI2PC Train] Testset samples: {len(val_loader)}")
    print(f"    >> [CSI2PC Train] Selected data split: {config['split_to_use']}")


    #############################
    #     DATA AUGMENTATION     #
    #############################

    data_augmentation = False
    print(f"    >> [CSI2PC Train] Data Augmentation: {data_augmentation}")

    if data_augmentation:
        # Delete the DataLoader object from memory
        del train_loader

        # If using GPU and you want to clear memory
        torch.cuda.empty_cache()

        augmented_train_data = create_augmented_dataset(train_dataset, num_augmentations=1)
        augmented_train_dataset = AugmentedMMFiDataset(augmented_train_data)

        train_loader = torch.utils.data.DataLoader(
            augmented_train_dataset,
            shuffle=True,
            drop_last=True,
            generator=rng_generator,
            batch_size=config['train_loader']['batch_size'],
            collate_fn=collate_fn_padd_csi2pc,
            num_workers=12,
            pin_memory=True
        )

    print(f"    >> [CSI2PC Train] Trainset samples (DA): {len(train_loader)}")
    print(f"    >> [CSI2PC Train] Testset samples (DA): {len(val_loader)}")



    #############################
    #        MODEL CONFIG       #
    #############################
    torch.cuda.empty_cache()

    # Initialize model, optimizer, loss function, and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # scaler = GradScaler()   # Use it if needed. FP16.

    model = CSI2PointCloudModel(
        embedding_dim=256,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_points=1200,  # Updated to match the fixed number of points
        num_antennas=3,
        num_subcarriers=114,
        num_time_slices=10
    ).to(device)

    learnig_rate = 1e-4
    optimizer_name = "NAdam"
    optimizer = torch.optim.NAdam(model.parameters(), lr=learnig_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # loss_fn = chamfer_distance
    loss_fn = CSI2PointCloudLoss(chamfer_weight=0.85, regularization_weight=0.15)

    print_model_settings(model)

    num_epochs = 30  # Set your desired number of epochs
    history_da = {'train_loss': [], 'val_loss': []}
    t0 = time.time()

    for epoch in range(num_epochs):
        print('    >> EPOCH %d/%d' % (epoch + 1, num_epochs))
        t1 = time.time()

        # Training
        train_loss = train_epoch(
            model=model,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch
        )

        # Scheduler step
        scheduler.step()

        # Optionally, print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"    >> Learning Rate after epoch {epoch + 1}: {current_lr:.6e}")

        # Validation
        val_loss = test_epoch(
            model=model,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn,
            epoch=epoch + 1,
            dataset_type="Validation",
            visualize=True
        )

        # Record losses
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)

        # Print epoch summary
        print('    >> EPOCH {}/{} \t train loss {:.6f} \t val loss {:.6f}'.format(epoch + 1, num_epochs, train_loss, val_loss))

        # Save checkpoint at the end of each epoch
        model_version = "arch01_v1_001"
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir="../models/checkpoints/", name_model=model_version)

        print(f"    >> Consumed time in Epoch {epoch + 1}: {time.time() - t1:.2f} seconds \n")

        # Saving the model.
    save_model = True
    if save_model:
        model_version = "arch01_v1_001"
        day_model = "091024"
        torch.save(model.state_dict(), f"../models/scsi2pc_model_{day_model}{model_version}_ep{num_epochs}_lr{learnig_rate}_{optimizer_name}.pth")
        print(f" >> Model saved with name: scsi2pc_model_{day_model}{model_version}_ep{num_epochs}_lr{learnig_rate}_{optimizer_name}.pth")

    print(f"Total training time: {(time.time() - t0) / 60:.2f} minutes")


if __name__ == '__main__':
    main()
