import torch
import json
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

train_path = '../data/train'
test_path = '../data/test'

train_dataset = PlanetDataset(train_path, None, True, 10, "min")
test_dataset = PlanetDataset(train_path, None, True, 10, "min")

weight_decays = [0, 1e-5, 1e-4, 1e-3]

# load a model pre-trained on COCO
model = MODELS['mobilenet']()

# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 2  # 1 class (planet) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

learning_rate = 1e-4

LOG = {}

optimizer = OPTIMIZERS['adam'](model, learning_rate, weight_decays[2])
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                               gamma=0.1)

train(model, optimizer, lr_scheduler, train_dataset, collate_fn, device, LOG, epochs=1)

check_accuracy(test_dataset, model, collate_fn, LOG, None, device)

log_path = './log.json'
with open(log_path, 'w') as f:
    logs = json.dumps(LOG)
    f.write(logs)
