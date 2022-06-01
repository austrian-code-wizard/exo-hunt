import torch
import torchvision
import torch.optim as optim
from dataset import PlanetDataset
from utils import train, check_accuracy, collate_fn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

USE_GPU = True
dtype = torch.float32 # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device {device}")

train_path = '../data/train'
test_path = '../data/test'
train_dataset = PlanetDataset(train_path, None, True, 10)
test_dataset = PlanetDataset(test_path, None, False)

# DEFINE THE PRETRAINED MODEL

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 2  # 1 class (planet) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

learning_rate = 1e-3

#optimizer = optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, optimizer, train_dataset, collate_fn, device, epochs=10)

check_accuracy(train_dataset, model, collate_fn, device)

check_accuracy(test_dataset, model, collate_fn, device)