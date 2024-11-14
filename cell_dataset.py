import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from pyntcloud import PyntCloud
from sklearn import preprocessing
from pathlib import Path


class PointCloudDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=64,
        label_col="Treatment",
        transform=None,
        target_transform=None,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
        ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        img_path = os.path.join(
            self.img_dir, treatment, self.new_df.loc[idx, "serialNumber"]
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.tensor([[13.4828, 26.5144, 24.4187]])
        std = torch.tensor([[9.2821, 20.4512, 18.9049]])
        image = (image - mean) / std
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, label, feats, serial_number


class PointCloudDatasetAll(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=64,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
        ].reset_index(drop=True)
        # ((self.annot_df.Treatment == 'Nocodazole') |
        # (self.annot_df.Treatment == 'Blebbistatin'))
        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.tensor([[13.4828, 26.5144, 24.4187]])
        std = torch.tensor([[9.2821, 20.4512, 18.9049]])
        image = (image - mean) / std
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, label, feats, serial_number


class VesselMNIST3D(Dataset):

    def __init__(self, points_dir, centre=True, scale=20.0, partition="train"):
        self.points_dir = points_dir
        self.centre = centre
        self.scale = scale
        self.p = Path(self.points_dir)
        self.partition = partition
        self.path = self.p / partition
        self.files = list(self.path.glob("**/*.ply"))
        self.classes = [x.parents[0].name.replace("_pointcloud", "") for x in self.files]

        self.le = preprocessing.LabelEncoder()
        self.class_labels = self.le.fit_transform(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        label = self.class_labels[idx]
        class_name = self.classes[idx]
        point_cloud = PyntCloud.from_file(str(file))
        mean = 0
        point_cloud = torch.tensor(point_cloud.points.values)
        if self.centre:
            mean = torch.mean(point_cloud, 0)

        scale = torch.tensor([[self.scale, self.scale, self.scale]])
        point_cloud = (point_cloud - mean) / scale
        pc = PCA(n_components=3)
        u = torch.tensor(pc.fit_transform(point_cloud.numpy()))

        return (
            point_cloud,
            torch.tensor(label, dtype=torch.int64),
            u,
            class_name,
        )