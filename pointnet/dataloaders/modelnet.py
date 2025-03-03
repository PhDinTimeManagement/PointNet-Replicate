from typing import List
import torch
import numpy as np
import h5py
import os
import os.path as osp
from utils.misc import pc_normalize


class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, phase: str, data_dir: str):
        super().__init__()
        self.phase = phase
        self.data_dir = data_dir
        self.modelnet_dir = osp.join(data_dir, "modelnet40_ply_hdf5_2048")

        self.download_data()

        # ModelNet has only train and test splits.
        if phase == "val":
            phase = "test"

        with open(osp.join(self.modelnet_dir, f"{phase}_files.txt")) as f:
            file_list = [line.rstrip() for line in f]

        self.data = []
        self.label = []
        self.normal = []
        for fn in file_list:
            f = h5py.File(osp.join(self.modelnet_dir, osp.basename(fn)))
            self.data.append(f["data"][:])
            self.label.append(f["label"][:])
            self.normal.append(f["normal"][:])
            # if "normal" in f.keys():
            #     self.normal.append(f["normal"][:])
            # else:
            #     print(f"Warning: 'normal' key not found in {fn}. Using zeros.")
            #     self.normal.append(np.zeros_like(f["data"][:]))  # Ensure correct shape

        self.data = np.concatenate(self.data, 0).astype(np.float32)
        self.label = np.concatenate(self.label, 0).astype(np.int_)
        self.normal = np.concatenate(self.normal, 0).astype(np.float32)

    def __getitem__(self, idx):
        pc = torch.from_numpy(pc_normalize(self.data[idx]))
        label = torch.from_numpy(self.label[idx]).squeeze()

        return pc, label

    def __len__(self):
        return len(self.data)

    def download_data(self):
        if not osp.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        if not osp.exists(self.modelnet_dir):
            # www = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
            # www = "https://onedrive.live.com/?authkey=%21ALKmMDfOhwxH43k&id=0CE615B143FC4BDC%21188223&cid=0CE615B143FC4BDC&parId=root&parQt=sharedby&o=OneUp"
            # www = "https://irl9ca.dm.files.1drv.com/y4mxhcxnXIkRYKvO4_iGfTiT4cfTbtOrp0SgT9EbSSeK5kHylx8q2lqhRgpelf3rXAoYkXjHBh6jTpwob--VuPv1_D8xNQYBFveDP_Bel7rT9u49CRhFEYSE6itsMIo4RWOn4eFaKWZupPKT7oOs8HGRFIwwK_Zdlth0VuJnU7QEyU_1t0o8ziFo4g4hJ_UToGr7agTBekgEAiQCyYPflmYyg"
            www = "https://cdn-lfs.hf.co/repos/36/90/3690071fa5fce49ab533e0e81bd5b7fd1fc7337b68386704b06e7c4dfe6eed96/f01b8189281fae5790e39deb9f3eca86e446b771bdc665c6ad05f28d039b20e7?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27modelnet40_ply_hdf5_2048.zip%3B+filename%3D%22modelnet40_ply_hdf5_2048.zip%22%3B&response-content-type=application%2Fzip&Expires=1740990798&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0MDk5MDc5OH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9yZXBvcy8zNi85MC8zNjkwMDcxZmE1ZmNlNDlhYjUzM2UwZTgxYmQ1YjdmZDFmYzczMzdiNjgzODY3MDRiMDZlN2M0ZGZlNmVlZDk2L2YwMWI4MTg5MjgxZmFlNTc5MGUzOWRlYjlmM2VjYTg2ZTQ0NmI3NzFiZGM2NjVjNmFkMDVmMjhkMDM5YjIwZTc%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=K0SFP0rfR9UBZ1L7GHtiiS8-Ls8Z00Oe%7EGFUTh9qv4TPraKrSzOe-scWxwqsGvDWEdLyNNH2Jv7SBxBXBF-2QZ0iycGhDCS1-6VJwMIjVryobEsGK3xk6ZYJL4I%7ECXnyZ%7EZvX5uMQ2Dbcj0PkifNX4jPhEMbuzIVT-VrQUH%7EbjrE4MQk2XkxR%7ElbVLJ-IeYPCT9-5QQPBAmtehVWrvgCs9FlZ7fZxrDICOuF0TMa8Ic88ciHl41UTTgfrv7iv6bQ49MjHXpr52PjTr703twpUURGfa9khXucyeft4p0qqSCBLYFkqMXpTzZsq2Y8luZ3GkRC6qxEbaCr1h1T1OpDtA__&Key-Pair-Id=K3RPWS32NSSJCE"
            zipfile = osp.basename(www)
            os.system(f"wget --no-check-certificate {www}; unzip {zipfile}")
            os.system(f"mv {zipfile[:-4]} {self.data_dir}")
            os.system(f"rm {zipfile}")


def get_data_loaders(
    data_dir, batch_size, phases: List[str] = ["train", "val", "test"]
):
    datasets = []
    dataloaders = []
    for ph in phases:
        ds = ModelNetDataset(ph, data_dir)
        dl = torch.utils.data.DataLoader(
            ds, batch_size, shuffle=ph == "train", drop_last=ph == "train"
        )

        datasets.append(ds)
        dataloaders.append(dl)

    return datasets, dataloaders
