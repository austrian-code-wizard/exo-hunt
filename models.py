import torchvision
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn as nn

# Returns the ResNet 50 backbone
def get_resnet_model():
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

# Returns the MobileNet V2 backbone
def get_mobilenet_model():
    backbone = torchvision.models.mobilenet_v2(pretrained=False).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=2, image_mean=[0.0006, 0.0006, 0.0006], image_std=[0.0250, 0.0246, 0.0240], rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

# Returns the MobileNet V2 backbone
def get_inception_model():
    backbone = torchvision.models.inception_v3(pretrained=False).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=2, image_mean=[0.0006, 0.0006, 0.0006], image_std=[0.0250, 0.0246, 0.0240], rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

# Returns the VGG 16 backbone
def get_vgg_model():
    backbone = torchvision.models.vgg16(pretrained=True).features
    backbone.out_channels = 512 
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

# Returns the ConvNextTiny backbone
def get_convnext_tiny_model():
    backbone = torchvision.models.convnext_tiny(pretrained=False).features
    backbone.out_channels = 768 
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=2, image_mean=[0.0006, 0.0006, 0.0006], image_std=[0.0250, 0.0246, 0.0240], rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

def get_3layer_model():
    size = (32 + 4 - 5) + 1
    size = (size + 2 - 3) + 1
    channel_1 = 32
    channel_2 = 16

    backbone = nn.Sequential(
        nn.Conv2d(9, 3, 1, padding=0),
        nn.ReLU(),
        nn.Conv2d(3, channel_1, 5, padding=2),
        nn.ReLU(),
        nn.Conv2d(channel_1, channel_2, 3, padding=1),
    )
    backbone.out_channels = 16 
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=2, image_mean=[-4.1760e-06, -4.2734e-06, -4.3148e-06, -4.3536e-06, -4.3248e-06,
        -4.2478e-06, -4.2522e-06, -4.1569e-06, -4.1071e-06], image_std=[0.0012, 0.0010, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008], rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

MODELS = {
    'resnet': get_resnet_model, 
    'mobilenet': get_mobilenet_model, 
    'vgg': get_vgg_model,
    'convnext-tiny': get_convnext_tiny_model,
    'inception': get_inception_model,
    '3layer': get_3layer_model
}
