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
            # stn3 returns a [B, 3, 3] transformation matrix.
            trans_3x3 = self.stn3(pointcloud)
            # Apply transform: [B, N, 3] x [B, 3, 3] -> [B, N, 3]
            pointcloud = torch.bmm(pointcloud, trans_3x3)

        # Reshape from [B, N, 3] to [B, 3, N] to use Conv1d as MLP
        # PyTorch requires shape [B, C, N]
        x = pointcloud.transpose(2, 1)

        # (2) First MLP layers (64, 64)
        x = self.mlp1(x) # [B, 3, N] -> [B, 64, N]

        # (3) Feature Transform (if enabled)
        if self.feature_transform:
            # stn64 returns a [B, 64, 64] transformation matrix.
            trand_64x64 = self.stn64(x)
            # [B, 64, N] -> [B, N, 64]
            x = x.transpose(2, 1)
            # Apply transform: [B, N, 64] x [B, 64, 64] -> [B, N, 64]
            x = torch.bmm(x, trand_64x64)
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
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        pass


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        pass

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        pass


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
