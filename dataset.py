import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from pyntcloud import PyntCloud
import numpy as np
import random


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=1, clip=0.02):
    N, C = pointcloud.shape
    rotation = np.copy(pointcloud)
    rotation += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return rotation


def generate_24_rotations():
    res = []
    for id in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        R = np.identity(3)[:, id].astype(int)
        R1= np.asarray([R[:, 0], R[:, 1], R[:, 2]]).T
        R2 = np.asarray([-R[:, 0], -R[:, 1], R[:, 2]]).T
        R3 = np.asarray([-R[:, 0], R[:, 1], -R[:, 2]]).T
        R4 = np.asarray([R[:, 0], -R[:, 1], -R[:, 2]]).T
        res += [R1, R2, R3, R4]
    for id in [[0, 2, 1], [1, 0, 2], [2, 1, 0]]:
        R = np.identity(3)[:, id].astype(int)
        R1 = np.asarray([-R[:, 0], -R[:, 1], -R[:, 2]]).T
        R2 = np.asarray([-R[:, 0], R[:, 1], R[:, 2]]).T
        R3 = np.asarray([R[:, 0], -R[:, 1], R[:, 2]]).T
        R4 = np.asarray([R[:, 0], R[:, 1], -R[:, 2]]).T
        res += [R1, R2, R3, R4]
    return res


def rotate_pointcloud(pointcloud):
    # theta = np.random.normal(0, (np.pi**2)/16, 1)[0]
    # print(theta)
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    rotation = np.copy(pointcloud)
    rotation[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return rotation, theta


def three_d_rotation(pointcloud):
    alpha = np.pi * 2 * np.random.choice(24) / 24
    beta = np.pi * 2 * np.random.choice(24) / 24
    gamma = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array(
        [[np.cos(beta) * np.cos(gamma),
          (np.sin(alpha) * np.sin(beta) * np.cos(gamma)) - (np.cos(alpha) * np.cos(gamma)),
          (np.cos(alpha) * np.sin(beta) * np.cos(gamma)) + (np.sin(alpha) * np.sin(gamma))],

         [np.cos(beta) * np.sin(gamma),
          (np.sin(alpha) * np.sin(beta) * np.sin(gamma)) + (np.cos(alpha) * np.cos(gamma)),
          (np.cos(alpha) * np.sin(beta) * np.sin(gamma)) - (np.sin(alpha) * np.cos(gamma))],

         [-np.sin(beta),
          np.sin(alpha) * np.cos(beta),
          np.cos(alpha) * np.cos(beta)]]
    )
    rotation = np.copy(pointcloud)
    rotation[:, ] = pointcloud[:, ].dot(rotation_matrix) 
    return rotation, (alpha, beta, gamma)


class PointCloudDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=False,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.centring_only = centring_only

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
        # TODO: take away after testing
        if self.centring_only:
            mean = torch.mean(image, 0)
            # mean = torch.tensor([[13.4828, 26.5144, 24.4187]])
            # std = torch.tensor([[9.2821, 20.4512, 18.9049]])
            std = torch.tensor([[20.0, 20.0, 20.0]])
            image = (image - mean) / std
        # / std
        # TODO: _____________________________________________
        else:
            mean = torch.tensor([[13.4828, 26.5144, 24.4187]])
            std = torch.tensor([[9.2821, 20.4512, 18.9049]])
            image = (image - mean) / std
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        return image, label, feats


class PointCloudDatasetAll(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=True,
        cell_component="cell",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.centring_only = centring_only
        self.cell_component = cell_component

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

        # TODO: take away after testing
        if self.centring_only:
            mean = torch.mean(image, 0)
            std = torch.tensor([[20.0, 20.0, 20.0]])
            image = (image - mean) / std

        else:
            mean = torch.tensor([[13.4828, 26.5144, 24.4187]])
            std = torch.tensor([[9.2821, 20.4512, 18.9049]])
            image = (image - mean) / std

        # TODO: _____________________________________________
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, label, serial_number


class PointCloudDatasetAllBoth(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=False,
        cell_component="cell",
        proximal=1,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.proximal = proximal

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
            & (
                (self.annot_df.Treatment == "Nocodazole")
                | (self.annot_df.Treatment == "Blebbistatin")
            )
            & (self.annot_df.Proximal == self.proximal)
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
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        cell_path = "stacked_pointcloud"
        nuc_path = "stacked_pointcloud_nucleus"

        cell_img_path = os.path.join(
            self.img_dir,
            plate_num,
            cell_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        nuc_img_path = os.path.join(
            self.img_dir,
            plate_num,
            nuc_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        cell = PyntCloud.from_file(cell_img_path + ".ply")
        nuc = PyntCloud.from_file(nuc_img_path + ".ply")

        cell = torch.tensor(cell.points.values)
        nuc = torch.tensor(nuc.points.values)
        full = torch.tensor(np.concatenate((cell[:1024], nuc[:1024])))
        mean = torch.mean(full, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])

        image = (full - mean) / std

        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        return image, label, feats


class PointCloudDatasetAllBothNotSpec(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=False,
        cell_component="cell",
        proximal=1,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.proximal = proximal

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
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        cell_path = "stacked_pointcloud"
        nuc_path = "stacked_pointcloud_nucleus"

        cell_img_path = os.path.join(
            self.img_dir,
            plate_num,
            cell_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        nuc_img_path = os.path.join(
            self.img_dir,
            plate_num,
            nuc_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        cell = PyntCloud.from_file(cell_img_path + ".ply")
        nuc = PyntCloud.from_file(nuc_img_path + ".ply")

        cell = torch.tensor(cell.points.values)
        nuc = torch.tensor(nuc.points.values)
        full = torch.tensor(np.concatenate((cell[:1024], nuc[:1024])))
        mean = torch.mean(full, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])

        image = (full - mean) / std

        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, label, serial_number


class PointCloudDatasetAllBothNotSpec1024(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=True,
        cell_component="cell",
        proximal=1,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.proximal = proximal

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
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        cell_path = "stacked_pointcloud"
        nuc_path = "stacked_pointcloud_nucleus"

        cell_img_path = os.path.join(
            self.img_dir,
            plate_num,
            cell_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        nuc_img_path = os.path.join(
            self.img_dir,
            plate_num,
            nuc_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        cell = PyntCloud.from_file(cell_img_path + ".ply")
        nuc = PyntCloud.from_file(nuc_img_path + ".ply")

        cell = torch.tensor(cell.points.values)
        nuc = torch.tensor(nuc.points.values)
        full = torch.tensor(np.concatenate((cell[:512], nuc[:512])))
        mean = torch.mean(full, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])

        image = (full - mean) / std

        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, label, serial_number


class PointCloudDatasetAll1024(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=True,
        cell_component="cell",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.centring_only = centring_only
        self.cell_component = cell_component

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

        # TODO: take away after testing
        if self.centring_only:
            image = image[:1024]
            mean = torch.mean(image, 0)
            std = torch.tensor([[20.0, 20.0, 20.0]])
            image = (image - mean) / std

        # TODO: _____________________________________________
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, label, serial_number


class PointCloudDatasetAllRotation1024(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=True,
        cell_component="cell",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.centring_only = centring_only
        self.cell_component = cell_component

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

        # TODO: take away after testing

        image = image[:1024]
        mean = torch.mean(image, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])
        image = (image - mean) / std
        rotated_image, angles = three_d_rotation(image)

        # TODO: _____________________________________________
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, rotated_image, angles, serial_number


class PointCloudDatasetAllBothNotSpecRotation(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=False,
        cell_component="cell",
        proximal=1,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.proximal = proximal

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
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        cell_path = "stacked_pointcloud"
        nuc_path = "stacked_pointcloud_nucleus"

        cell_img_path = os.path.join(
            self.img_dir,
            plate_num,
            cell_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        nuc_img_path = os.path.join(
            self.img_dir,
            plate_num,
            nuc_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        cell = PyntCloud.from_file(cell_img_path + ".ply")
        nuc = PyntCloud.from_file(nuc_img_path + ".ply")

        cell = torch.tensor(cell.points.values)
        nuc = torch.tensor(nuc.points.values)
        full = torch.tensor(np.concatenate((cell[:1024], nuc[:1024])))
        mean = torch.mean(full, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])

        image = (full - mean) / std
        rotated_image, angles = three_d_rotation(image.numpy())
        rotated_image = torch.tensor(rotated_image)
        angles = torch.tensor(angles)

        # TODO: _____________________________________________
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, rotated_image, angles, serial_number

    
class PointCloudDatasetAllBothNotSpecRotation1024(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=True,
        cell_component="cell",
        proximal=1,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.proximal = proximal

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
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        cell_path = "stacked_pointcloud"
        nuc_path = "stacked_pointcloud_nucleus"

        cell_img_path = os.path.join(
            self.img_dir,
            plate_num,
            cell_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        nuc_img_path = os.path.join(
            self.img_dir,
            plate_num,
            nuc_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        cell = PyntCloud.from_file(cell_img_path + ".ply")
        nuc = PyntCloud.from_file(nuc_img_path + ".ply")

        cell = torch.tensor(cell.points.values)
        nuc = torch.tensor(nuc.points.values)
        full = torch.tensor(np.concatenate((cell[:512], nuc[:512])))
        mean = torch.mean(full, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])

        image = (full - mean) / std
        rotated_image, angles = three_d_rotation(image.numpy())
        rotated_image = torch.tensor(rotated_image)
        angles = torch.tensor(angles)

        # TODO: _____________________________________________
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, rotated_image, angles, serial_number


class PointCloudDatasetAllBothNotSpec2DRotation1024(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=True,
        cell_component="cell",
        proximal=1,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.proximal = proximal

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
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        cell_path = "stacked_pointcloud"
        nuc_path = "stacked_pointcloud_nucleus"

        cell_img_path = os.path.join(
            self.img_dir,
            plate_num,
            cell_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        nuc_img_path = os.path.join(
            self.img_dir,
            plate_num,
            nuc_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        cell = PyntCloud.from_file(cell_img_path + ".ply")
        nuc = PyntCloud.from_file(nuc_img_path + ".ply")

        cell = torch.tensor(cell.points.values)
        nuc = torch.tensor(nuc.points.values)
        full = torch.tensor(np.concatenate((cell[:512], nuc[:512])))
        mean = torch.mean(full, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])

        image = (full - mean) / std
        rotated_image, angles = rotate_pointcloud(image.numpy())
        rotated_image = torch.tensor(rotated_image)
        angles = torch.tensor(angles)

        # TODO: _____________________________________________
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, rotated_image, angles, serial_number


class PointCloudDatasetAllBothKLDivergranceRotation1024(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=True,
        cell_component="cell",
        proximal=1,
        rotation_matrices=generate_24_rotations(),
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.proximal = proximal
        self.rotation_matrices = rotation_matrices

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
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        cell_path = "stacked_pointcloud"
        nuc_path = "stacked_pointcloud_nucleus"

        cell_img_path = os.path.join(
            self.img_dir,
            plate_num,
            cell_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        nuc_img_path = os.path.join(
            self.img_dir,
            plate_num,
            nuc_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        cell = PyntCloud.from_file(cell_img_path + ".ply")
        nuc = PyntCloud.from_file(nuc_img_path + ".ply")

        cell = torch.tensor(cell.points.values)
        nuc = torch.tensor(nuc.points.values)
        full = torch.tensor(np.concatenate((cell[:512], nuc[:512])))
        mean = torch.mean(full, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])

        image = (full - mean) / std
        rotation_matrix = torch.tensor(self.rotation_matrices[random.randrange(0, 24)]).type(torch.FloatTensor)
        rotated_image = torch.matmul(image, rotation_matrix)

        # TODO: _____________________________________________
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, rotated_image, serial_number


class SimCLR1024Both(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        centring_only=True,
        cell_component="cell",
        proximal=1,
        rotation_matrices=generate_24_rotations(),
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.proximal = proximal
        self.rotation_matrices = rotation_matrices

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
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        cell_path = "stacked_pointcloud"
        nuc_path = "stacked_pointcloud_nucleus"

        cell_img_path = os.path.join(
            self.img_dir,
            plate_num,
            cell_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        nuc_img_path = os.path.join(
            self.img_dir,
            plate_num,
            nuc_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )

        cell = PyntCloud.from_file(cell_img_path + ".ply")
        nuc = PyntCloud.from_file(nuc_img_path + ".ply")

        cell = torch.tensor(cell.points.values)
        nuc = torch.tensor(nuc.points.values)
        full = torch.tensor(np.concatenate((cell[:512], nuc[:512])))
        mean = torch.mean(full, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])

        image = (full - mean) / std
        rotation_matrix = torch.tensor(
            self.rotation_matrices[random.randrange(1, 24)]).type(torch.FloatTensor)
        rotated_image = torch.matmul(image, rotation_matrix)
        rotated_jitter = jitter_pointcloud(rotated_image)
        rotated_jitter_translated = translate_pointcloud(rotated_jitter)

        rotation_matrix2 = torch.tensor(
            self.rotation_matrices[random.randrange(1, 24)]).type(torch.FloatTensor)
        rotated_image2 = torch.matmul(rotated_image, rotation_matrix2)
        rotated_jitter2 = jitter_pointcloud(rotated_image2)
        rotated_jitter_translated2 = translate_pointcloud(rotated_jitter2)

        # TODO: _____________________________________________
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, rotated_jitter_translated, rotated_jitter_translated2, serial_number
