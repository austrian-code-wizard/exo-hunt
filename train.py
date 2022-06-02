import torch
from dataset import PlanetDataset
from models import MODELS 
from optimizers import OPTIMIZERS
from utils import train, check_accuracy, collate_fn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

USE_GPU = True
dtype = torch.float32 # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device {device}")

train_path = '../data2/train'
test_path = '../data2/test'

train_dataset = PlanetDataset(train_path, None, True, 300, "conv")
test_dataset = PlanetDataset(test_path, None, False, 100, "conv")

weight_decays = [0, 1e-5, 1e-4, 1e-3]

# load a model pre-trained on COCO
model = MODELS['3layer']()

# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 2  # 1 class (planet) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

learning_rate = 5e-5


optimizer = OPTIMIZERS['adam'](model, learning_rate, weight_decays[1])
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1,
                                               gamma=0.1)

train(model, optimizer, lr_scheduler, train_dataset, collate_fn, device, epochs=3)

check_accuracy(test_dataset, model, collate_fn, device)