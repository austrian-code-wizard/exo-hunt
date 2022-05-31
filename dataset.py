import torch
import math
import numpy as np
from astropy.io import fits
from PIL import Image
from labels import LABELS
from pathlib import Path


def deg_to_box(deg, sep, radius=3):
    deg += 90
    deg = math.radians(deg)
    y_center = -math.sin(deg) * sep
    x_center = math.cos(deg) * sep
    y_center += 451 // 2
    x_center += 451 // 2
    return int(x_center - radius), int(y_center - radius), int(x_center + radius), int(y_center + radius)


class PlanetDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, train=True):
        self.root = root
        self.transforms = transforms
        self.train = train
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = []
        self.labels = []
        for img in Path(root).rglob("*.fits"):
            path = str(img.absolute()).split("/")
            if path[-1].split("_")[0] not in LABELS:
                continue
            obj = path[-3]
            date = path[-2]
            self.imgs.append(img.absolute())
            self.labels.append(LABELS[obj][date])

    def __getitem__(self, idx):
        # load images and masks
        data = fits.getdata(self.imgs[idx])
        data = np.nan_to_num(data)
        data = np.mean([data[:3,:,:], data[3:6,:,:], data[6:,:,:]], axis=0)
        data = np.transpose(data, (1, 2, 0))

        img = Image.fromarray((data * 255).astype(np.uint8)).convert("RGB")
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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)