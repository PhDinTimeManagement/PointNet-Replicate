from hmac import trans_36

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    Extract global features from point clouds.
    """
    def __init__(
            self,
            input_transform: bool = False,
            feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.

        # Input Transform (T-Net 3x3)
        if self.input_transform:
            self.stn3 = STNKd(k=3)

        # MLP layers (64, 64)
        # To achieve point-wise feature transform, let kernel size be 1.
        self.mlp1 = nn.Sequential(
            # Convert each point's (x, y, z) 3D input to 64-dimensional features.
            nn.Conv1d(3, 64, 1), # (B, 3, N) -> (B, 64, N)
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Another transformation to refine the 64-dimensional features.
            nn.Conv1d(64, 64, 1), # (B, 64, N) -> (B, 64, N)
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Feature Transform (T-Net 64x64)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # MLP layers (64, 128, 1024)
        self.mlp2 = nn.Sequential(
            # (B, 64, N) -> (B, 64, N) is not necessary, because it's already done in mlp1.

            # Convert each point's 64-dimensional features to 128-dimensional features.
            nn.Conv1d(64, 128, 1),  #[B, 64, N] -> [B, 128, N]
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Convert each point's 128-dimensional features to 1024-dimensional features.
            nn.Conv1d(128, 1024, 1), #[B, 128, N] -> [B, 1024, N]
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
            - [batch, num_points, xyz]
        Output:
            - Global feature: [B,1024]
            - FloatTensor of shape [batch, 1024]
        """

        # TODO : Implement forward function.

        # Get the shape of the input point cloud
        B, N, _ = pointcloud.shape

        # (1) Input Transform
        if self.input_transform:
            # Transpose the input point cloud to [B, 3, N] so stn3 sees 3 channels
            pc_for_tent = pointcloud.transpose(2, 1) # [B, N, 3] -> [B, 3, N]
            trans_3x3 = self.stn3(pc_for_tent) # stn3 forward pass -> [B, 3, 3]

            # Multiply [B, N, 3] x [B, 3, 3] to transform to the original cloud
            pointcloud = torch.bmm(pointcloud, trans_3x3)

        # Reshape from [B, N, 3] to [B, 3, N] to use Conv1d as MLP
        # PyTorch requires shape [B, C, N]
        x = pointcloud.transpose(2, 1)

        # (2) First MLP layers (64, 64)
        x = self.mlp1(x) # [B, 3, N] -> [B, 64, N]

        # (3) Feature Transform
        if self.feature_transform:
            # stn64 returns a [B, 64, 64] transformation matrix.
            trans_64x64 = self.stn64(x)
            # [B, 64, N] -> [B, N, 64]
            x = x.transpose(2, 1)
            # Apply transform: [B, N, 64] x [B, 64, 64] -> [B, N, 64]
            x = torch.bmm(x, trans_64x64)
            # Back to [B, 64, N]
            x = x.transpose(2, 1)

        # (4) Second MLP layers (64, 128, 1024)
        x = self.mlp2(x) # [B, 64, N] -> [B, 1024, N]

        # (5) Max Pooling
        global_feature = torch.max(x, 2)[0] # [B, 1024, N] -> [B, 1024]

        return global_feature # [B, 1024]


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        # Feature extractor (produces 1024-dimensional global feature)
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.

        # Classification MLP (1024 -> 512 -> 256 -> num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), # [B, 1024] -> [B, 512]
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256), # [B, 512] -> [B, 256]
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Dropout(0.3), # Dropout layer before the final layer

            # Final logits layer
            nn.Linear(256, num_classes) # [B, 256] -> [B, num_classes]
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
            - [batch, num_points, xyz]
        Output:
            - logits [B,num_classes]
            - classifcation scores for each class
        """

        # TODO : Implement forward function.

        # Extract 1024-dimensional global feature
        global_feature = self.pointnet_feat(pointcloud) # [B, 1024]

        # Compute logits for classification
        logits = self.mlp(global_feature) # [B, num_classes]

        return logits


class PointNetPartSeg(nn.Module):
    """
       PointNet for part segmentation
       Outouts per-point logits of shape [B, n, N]
       Combines local + global features:
        - Local features: [B, 64, N]
        - Global features: [B, 1024] (then repeated for each point)
        => Concatenated features: [B, 1088, N]
       Then use MLP maps 1088 -> 512 ->256 -> 128 -> m
       """
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.

        # 1) Input transform (T-Net 3x3)
        self.stn3 = STNKd(k=3)

        # 2) MLP layers (3 -> 64 -> 64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), # [B, 3, N] -> [B, 64, N]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1), # [B, 64, N] -> [B, 64, N]
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 3) Feature transform (T-Net 64x64)
        self.stn64 = STNKd(k=64)

        # 4) MLP layers (64 -> 128 -> 1024), this is used only for global features extraction
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), # [B, 64, N] -> [B, 128, N]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1), # [B, 128, N] -> [B, 1024, N]
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # 5) Final Segmentation MLP (1088 -> 512 -> 256 -> 128 -> m)
        self.mlp3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1), # [B, 1088, N] -> [B, 512, N]
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1), # [B, 512, N] -> [B, 256, N]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1), # [B, 256, N] -> [B, 128, N]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, m, 1) # [B, 128, N] -> [B, m, N]
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3] -> [batch, num_points, xyz]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """

        # TODO: Implement forward function.

        B, N, _ = pointcloud.shape

        # 1) Input transform (T-Net 3x3)
        # T-Net expects [B, 3, N] input, we multiply with [B, N, 3]
        pc_t = pointcloud.transpose(2, 1) # [B, N, 3] -> [B, 3, N]
        trans_3x3 = self.stn3(pc_t) # [B, 3, N] -> [B, 3, 3]
        pointcloud = torch.bmm(pointcloud, trans_3x3) # [B, N, 3] x [B, 3, 3] -> [B, N, 3]

        # 2) First MLP layers for local features (64, 64)
        x = pointcloud.transpose(2, 1) # [B, N, 3] -> [B, 3, N]
        x = self.mlp1(x)

        # 3) Feature transform (T-Net 64x64)
        trans_64x64 = self.stn64(x) # [B, 64, N] -> [B, 64, 64]
        x = x.transpose(2, 1) # [B, 64, N] -> [B, N, 64]
        x = torch.bmm(x, trans_64x64) # [B, N, 64] x [B, 64, 64] -> [B, N, 64]
        x = x.transpose(2, 1) # [B, N, 64] -> [B, 64, N]
        local_feat = x # Save local features from this stage, [B, 64, N]

        # 4) Second MLP layers for global features (64, 128, 1024)
        x = self.mlp2(x) # [B, 64, N] -> [B, 1024, N]
        global_feat = torch.max(x, 2)[0] # [B, 1024, N] -> [B, 1024]

        # 5) Concatenate local and global features [B, 1088, N]
        global_feat_expand = global_feat.unsqueeze(-1).repeat(1, 1, N) # [B, 1024] -> [B, 1024, N]
        seg_input = torch.cat([local_feat, global_feat_expand], dim=1) # [B, 64, N] + [B, 1024, N] -> [B, 1088, N]

        # 6) Final Segmentation MLP (1088 -> 512 -> 256 -> 128 -> m)
        logits = self.mlp3(seg_input) # [B, 1088, N] -> [B, m, N]

        return logits


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        pass


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
