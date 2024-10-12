# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2024, Constantino Álvarez, Tuomas Määttä, Sasan Sharifipour, Miguel Bordallo  (CMVS - University of Oulu)
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        # src shape: [batch_size, sequence_length, embedding_dim]
        output = self.transformer_encoder(src)
        return output  # Shape: [batch_size, sequence_length, embedding_dim]


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, num_points):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Initialize learnable query embeddings
        self.query_embeddings = nn.Parameter(torch.randn(num_points, embedding_dim))

    def forward(self, memory):
        # memory: Encoder outputs, shape [batch_size, sequence_length, embedding_dim]
        batch_size = memory.size(0)
        # Expand query embeddings to match batch size
        tgt = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, num_points, embedding_dim]
        output = self.transformer_decoder(tgt, memory)  # Shape: [batch_size, num_points, embedding_dim]
        return output


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: [batch_size * num_features, in_channels, sequence_length]
        x = self.conv(x)  # Shape: [batch_size * num_features, out_channels, sequence_length]
        x = F.relu(x)  # Apply activation
        x = x.mean(dim=2)  # Average over the sequence_length dimension
        # Now x has shape: [batch_size * num_features, out_channels]
        return x


class OutputProjection(nn.Module):
    def __init__(self, embedding_dim):
        super(OutputProjection, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

    def forward(self, point_embeddings):
        # point_embeddings shape: [batch_size, num_points, embedding_dim]
        output_points = self.mlp(point_embeddings)  # Shape: [batch_size, num_points, 3]
        return output_points


class CSISTNkd(nn.Module):
    def __init__(self, k=64):
        super(CSISTNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))  # [batchsize, 128, n_pts]
        x = F.relu(self.bn2(self.conv2(x)))  # [batchsize, 256, n_pts]
        x = F.relu(self.bn3(self.conv3(x)))  # [batchsize, 1024, n_pts]
        x = torch.max(x, 2, keepdim=True)[0]  # [batchsize, 1024, 1]
        x = x.view(-1, 1024)  # [batchsize, 1024]

        x = F.relu(self.bn4(self.fc1(x)))  # [batchsize, 512]
        x = F.relu(self.bn5(self.fc2(x)))  # [batchsize, 256]
        x = self.fc3(x)  # [batchsize, k * k]

        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)  # Identity matrix
        x = x + iden
        x = x.view(-1, self.k, self.k)  # [batchsize, k, k]
        return x


# Improved PointNet??
class CSI2PointNetDecoder(nn.Module):
    def __init__(self, embedding_dim, feature_transform=False):
        super(CSI2PointNetDecoder, self).__init__()
        self.feature_transform = feature_transform
        self.stn = CSISTNkd(k=embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, 64, 1)
        self.conv5 = nn.Conv1d(64, 3, 1)  # Output 3D coordinates

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

        # Residual connections
        self.residual_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),  # Match the dimension to the main path
            nn.LeakyReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, point_embeddings):
        # point_embeddings: [batch_size, num_points, embedding_dim]
        x = point_embeddings.permute(0, 2, 1)  # [batch_size, embedding_dim, num_points]
        batchsize = x.size(0)
        n_pts = x.size(2)

        if self.feature_transform:
            trans_feat = self.stn(x)
            x = x.transpose(2, 1)  # [batchsize, num_points, embedding_dim]
            x = torch.bmm(x, trans_feat)  # Apply feature transformation
            x = x.transpose(2, 1)  # [batchsize, embedding_dim, num_points]
        else:
            trans_feat = None

        # PointNet Layers with BatchNorm and ReLU
        x = F.leaky_relu(self.bn1(self.conv1(x)))  # [batchsize, 512, num_points]
        x = F.leaky_relu(self.bn2(self.conv2(x)))  # [batchsize, 256, num_points]

        residual = self.residual_fc(point_embeddings)  # [batchsize, num_points, 128]
        residual = residual.permute(0, 2, 1)  # Match dimensions for addition with x

        x = F.leaky_relu(self.bn3(self.conv3(x)) + residual)  # [batchsize, 128, num_points]
        x = F.leaky_relu(self.bn4(self.conv4(x)))  # [batchsize, 64, num_points]
        x = self.conv5(x)  # [batchsize, 3, num_points]
        x = x.permute(0, 2, 1)  # [batchsize, num_points, 3]

        return x, trans_feat


class CSI2PointCloudModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_encoder_layers, num_decoder_layers, num_points,
                 num_antennas=3, num_subcarriers=114, num_time_slices=10):
        super(CSI2PointCloudModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_points = num_points
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.num_time_slices = num_time_slices
        self.num_heads = num_heads

        # Input Encoding
        # self.linear_proj = nn.Linear(num_time_slices * 2, embedding_dim)
        self.temporal_encoder = TemporalEncoder(in_channels=2, out_channels=embedding_dim)

        # Positional Encodings
        self.antenna_embeddings = nn.Embedding(num_antennas, embedding_dim)
        self.subcarrier_embeddings = nn.Embedding(num_subcarriers, embedding_dim)

        # Transformer Encoder
        self.encoder = TransformerEncoder(embedding_dim, num_heads, num_encoder_layers)

        # Transformer Decoder
        self.decoder = TransformerDecoder(embedding_dim, num_heads, num_decoder_layers, num_points)

        # Output Projection
        self.output_proj = OutputProjection(embedding_dim)

    def forward(self, wifi_csi_frame):
        # wifi_csi_frame shape: [batch_size, num_antennas, num_subcarriers, 2, num_time_slices]
        batch_size = wifi_csi_frame.size(0)
        num_antennas = self.num_antennas
        num_subcarriers = self.num_subcarriers
        num_time_slices = self.num_time_slices

        num_features = self.num_antennas * self.num_subcarriers  # 342

        # Rearrange dimensions to bring time and channels to the correct positions
        csi_data = wifi_csi_frame.permute(0, 1, 2, 4, 3)  # [batch_size, 3, 114, 10, 2]

        # Reshape to merge antennas and subcarriers into features
        csi_data = csi_data.reshape(batch_size, num_features, self.num_time_slices, 2)  # [batch_size, 342, 10, 2]

        # Permute to have channels first for Conv1d
        csi_data = csi_data.permute(0, 1, 3, 2)  # [batch_size, 342, 2, 10]

        # Reshape to merge batch_size and num_features
        csi_data = csi_data.reshape(batch_size * num_features, 2, self.num_time_slices)  # [batch_size * 342, 2, 10]

        # Apply Temporal Encoder
        temporal_features = self.temporal_encoder(csi_data)  # [batch_size * 342, embedding_dim]

        # Reshape back to [batch_size, num_features, embedding_dim]
        embeddings = temporal_features.view(batch_size, num_features, self.embedding_dim)  # [batch_size, 342, embedding_dim]

        # Generate Positional Encodings
        device = wifi_csi_frame.device

        # Antenna and subcarrier positional encodings
        antenna_indices = torch.arange(num_antennas, device=device).unsqueeze(1).expand(-1, num_subcarriers).reshape(-1)
        subcarrier_indices = torch.arange(num_subcarriers, device=device).unsqueeze(0).expand(num_antennas, -1).reshape(-1)

        antenna_encodings = self.antenna_embeddings(antenna_indices)  # Shape: [342, embedding_dim]
        subcarrier_encodings = self.subcarrier_embeddings(subcarrier_indices)  # Shape: [342, embedding_dim]

        # Sum antenna and subcarrier encodings
        positional_encodings = antenna_encodings + subcarrier_encodings  # Shape: [342, embedding_dim]

        # Expand positional encodings to match batch size
        positional_encodings = positional_encodings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, 342, embedding_dim]

        # Add positional encodings to embeddings
        embeddings = embeddings + positional_encodings  # Shape: [batch_size, 342, embedding_dim]

        # Transformer Encoder
        encoder_output = self.encoder(embeddings)  # Shape: [batch_size, 342, embedding_dim]

        # Transformer Decoder
        point_embeddings = self.decoder(encoder_output)  # Shape: [batch_size, num_points, embedding_dim]

        # Output Projection
        output_points = self.output_proj(point_embeddings)  # Shape: [batch_size, num_points, 3]

        return output_points, encoder_output
