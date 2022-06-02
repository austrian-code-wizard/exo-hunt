from telnetlib import GA
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

train_path = '../data2/train'
test_path = '../data2/test'

weight_decays = [0, 1e-5, 1e-4, 1e-3]

# SET HYPERPARAMETERS HERE
EXPERIMENT_NAME="test"
PRETRAINED_BACKBONE=False
EPOCHS = 3
WEIGHT_DECAY = weight_decays[2]
LEARNING_RATE = 1e-4
STEP_SIZE = 3
GAMMA = 0.1
MODEL = 'mobilenet'
OPTIMIZER = 'adam'
DIM_REDUCTION = 'conv'

# load dataset
train_dataset = PlanetDataset(train_path, None, True, 10, DIM_REDUCTION)
test_dataset = PlanetDataset(test_path, None, False, 10, DIM_REDUCTION)

# load a model pre-trained on COCO
model = MODELS[MODEL](DIM_REDUCTION, PRETRAINED_BACKBONE)

# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 2  # 1 class (planet) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


LOG = {
    'lr': LEARNING_RATE, 
    'model': MODEL,
    'dim_reduction': DIM_REDUCTION,
    'pretrained_backbone': PRETRAINED_BACKBONE,
    'epochs': EPOCHS, 
    'weight_decay': WEIGHT_DECAY, 
    'step_size': STEP_SIZE,
    'gamma': GAMMA,
    'optimizer': OPTIMIZER,
    'scheduler': 'stepLR',
    'test': {
        'final_test_accuracy': 0,
        'img_boxes': [],
    },
    # each entry contains loss, validation acc, time of training, batch size (meaning number of images trained in that epoch), and # of val training images
    'validation': [],
}

optimizer = OPTIMIZERS[OPTIMIZER](model, LEARNING_RATE, WEIGHT_DECAY)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=STEP_SIZE,
                                               gamma=GAMMA)

train(model, optimizer, lr_scheduler, train_dataset, collate_fn, device, LOG, epochs=EPOCHS)

check_accuracy(test_dataset, model, collate_fn, LOG, None, device)

log_path = f'./results/{EXPERIMENT_NAME}.json'
with open(log_path, 'w') as f:
    logs = json.dumps(LOG, indent=2)
    f.write(logs)
