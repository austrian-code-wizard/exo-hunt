import torchvision
import torch.nn as nn

# Returns the ResNet 50 backbone
def get_resnet_model():
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Returns the MobileNet Large 320 backbone
def get_mobilenet_model():
    return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

def get_3layer_model():
    size = (32 + 4 - 5) + 1
    size = (size + 2 - 3) + 1
    channel_1 = 32
    channel_2 = 16

    model = nn.Sequential(
        nn.Conv2d(9, channel_1, 5, padding=2),
        nn.ReLU(),
        nn.Conv2d(channel_1, channel_2, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(810000, 2)
    )

    return model

MODELS = {
    'resnet': get_resnet_model, 
    'mobilenet': get_mobilenet_model, 
    # '3layer': get_3layer_model
}