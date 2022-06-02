from random import shuffle
import torch
import math
import numpy as np
from astropy.io import fits
from PIL import Image
from labels import LABELS
from pathlib import Path
from torchvision import transforms
import torch.nn as nn


def deg_to_box(deg, sep, radius=3):
    deg += 90
    deg = math.radians(deg)
    y_center = -math.sin(deg) * sep
    x_center = math.cos(deg) * sep
    y_center += 451 // 2
    x_center += 451 // 2
    return int(x_center - radius), int(y_center - radius), int(x_center + radius), int(y_center + radius)


class PlanetDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, train=True, limit=None, reduction="mean"):
        self.root = root
        self.transforms = transforms
        self.train = train
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = []
        self.labels = []
        self.lim = limit
        self.reduction = reduction
        self.preModel = nn.Conv2d(9, 3, kernel_size=3, stride=1, padding=1)
        for object_dir in Path(root).glob("*"):
            for date_dir in object_dir.glob("*"):
                for file_dir in date_dir.rglob("*.fits"):
                    path = str(file_dir.absolute()).split("/")
                    if path[-1].split("_")[0] not in LABELS:
                        continue
                    obj = path[-3]
                    date = path[-2]
                    self.imgs.append(file_dir.absolute())
                    self.labels.append(LABELS[obj][date])

        if self.lim is not None:
            img_and_labels = [(img, lab) for img, lab in zip(self.imgs, self.labels)]
            shuffle(img_and_labels)
            self.imgs = [d[0] for d in img_and_labels][:self.lim]
            self.labels = [d[1] for d in img_and_labels][:self.lim]

    def __getitem__(self, idx):
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        # instances are encoded as different colors
        label = self.labels[idx]

        # get bounding box coordinates for each mask
        num_objs = len(label["thetas"])
        boxes = []
        for i in range(num_objs):
            xmin, ymin, xmax, ymax = deg_to_box(label["thetas"][i], label["seps"][i])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # load images and masks
        data = fits.getdata(self.imgs[idx])
        data = np.nan_to_num(data)
        if self.reduction == "mean":
            data = np.mean([data[:3,:,:], data[3:6,:,:], data[6:,:,:]], axis=0)
        elif self.reduction == "max":
            data = np.max([data[:3,:,:], data[3:6,:,:], data[6:,:,:]], axis=0)
        elif self.reduction == "min":
            data = np.min([data[:3,:,:], data[3:6,:,:], data[6:,:,:]], axis=0)
        elif self.reduction == "sum":
            data = np.sum([data[:3,:,:], data[3:6,:,:], data[6:,:,:]], axis=0)
        
        data = np.transpose(data, (1, 2, 0))

        if self.reduction != "conv":
            data = Image.fromarray((data * 255).astype(np.uint8)).convert("RGB")

        # convert img to tensor
        trans = transforms.Compose([transforms.ToTensor()])
        data = trans(data.astype(np.float32)).type(torch.FloatTensor)

        if self.transforms is not None:
            data, target = self.transforms(data, target)

        return data, target

    def __len__(self):
        return len(self.imgs)
