from torch.utils.data import Dataset
import math

def deg_to_box(deg, sep):
  y_center = math.sin(deg) * sep
  x_center = math.cos(deg) * sep
  y_center += 451 // 2
  x_center += 451 // 2
  return int(x_center - 3), int(y_center - 3), int(x_center + 3), int(y_center + 3)

class PlanetDataset(Dataset):
    def __init__(self, imgs, coords):
        self.data = imgs
        self.coords = coords

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) - 1

    def __getitem__(self, idx):
        pass
        # should return an img and target (which contains label, bounding box info, etc.)
