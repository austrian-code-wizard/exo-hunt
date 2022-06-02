import torchvision
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn as nn

mean_and_std = {'mean': {'mean': [0.0005585729377344251,
                                  0.0005360489594750106,
                                  0.000522407703101635],
                         'std': [0.023455360904335976, 0.02298579178750515, 0.022693036124110222]},
                'min': {'mean': [0.0007232138304971159,
                                 0.0006895358674228191,
                                 0.0006619797204621136],
                        'std': [0.026718422770500183, 0.026104681193828583, 0.025579839944839478]},
                'max': {'mean': [0.00042931686039082706,
                                 0.00042628523078747094,
                                 0.00042954040691256523],
                        'std': [0.020494040101766586, 0.020446548238396645, 0.020535748451948166]},
                'conv': {'mean': [-6.894125363032799e-06,
                                  -7.076187102939002e-06,
                                  -7.068654213071568e-06,
                                  -7.006037776591256e-06,
                                  -6.969607511564391e-06,
                                  -6.851818852737779e-06,
                                  -6.748939995304681e-06,
                                  -6.651837338722544e-06,
                                  -6.570968253072351e-06],
                         'std': [0.0011891546892002225,
                                 0.0009794622892513871,
                                 0.0009188703261315823,
                                 0.0009028696222230792,
                                 0.0008922165143303573,
                                 0.0008747951942496002,
                                 0.0008631845703348517,
                                 0.0008524955483153462,
                                 0.0008498522220179439]},
                'sum': {'mean': [0.00604355288669467,
                                 0.005914066918194294,
                                 0.005796901881694794],
                        'std': [0.07719344645738602, 0.0763750746846199, 0.07561986148357391]}}


def get_model(backbone, out_channels, dim_reduction):
    if dim_reduction == "conv":
        backbone = nn.Sequential(
            nn.Conv2d(9, 3, 1),
            nn.ReLU(),
            backbone
        )
    backbone.out_channels = out_channels
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=2, image_mean=mean_and_std[dim_reduction]["mean"],
                       image_std=mean_and_std[dim_reduction]["std"], rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model


# Returns the ResNet 50 backbone
def get_resnet_model(dim_reduction, pretrained=False):
    backbone = torchvision.models.resnet50(pretrained=pretrained).features
    return get_model(backbone, 1280, dim_reduction)


# Returns the MobileNet V2 backbone
def get_mobilenet_model(dim_reduction, pretrained=False):
    backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
    return get_model(backbone, 1280, dim_reduction)


# Returns the MobileNet V2 backbone
def get_inception_model(dim_reduction, pretrained=False):
    backbone = torchvision.models.inception_v3(pretrained=pretrained).features
    return get_model(backbone, 1280, dim_reduction)


# Returns the VGG 16 backbone
def get_vgg_model(dim_reduction, pretrained=False):
    backbone = torchvision.models.vgg16(pretrained=pretrained).features
    return get_model(backbone, 512, dim_reduction)


# Returns the ConvNextTiny backbone
def get_convnext_tiny_model(dim_reduction, pretrained=False):
    backbone = torchvision.models.convnext_tiny(pretrained=pretrained).features
    return get_model(backbone, 768, dim_reduction)

# Returns custom 3layer model


def get_3layer_model(dim_reduction, pretrained=False):
    assert not pretrained, "Cannot load weights for custom model"

    channel_1 = 64
    channel_2 = 32
    channel_out = 32
    
    """backbone = nn.Sequential(
        nn.Conv2d(3, channel_1, 7, padding=3),
        nn.BatchNorm2d(channel_1),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(channel_1, channel_2, 3, padding=1),
        nn.BatchNorm2d(channel_2),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(channel_2, channel_out, 3, padding=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),"""
    channel_out = 16
    drop_rate = 0.1

    backbone = nn.Sequential(
        nn.Conv2d(3, channel_2, 5, padding='same'),
        nn.BatchNorm2d(channel_2),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(p=drop_rate),
        nn.Conv2d(channel_2, channel_2, 3, padding='same'),
        nn.BatchNorm2d(channel_2),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(p=drop_rate),
        nn.Conv2d(channel_2, channel_out, 3, padding='same'),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(p=drop_rate),
    )
    return get_model(backbone, channel_out, dim_reduction)


MODELS = {
    'resnet': get_resnet_model,
    'mobilenet': get_mobilenet_model,
    'vgg': get_vgg_model,
    'convnext-tiny': get_convnext_tiny_model,
    'inception': get_inception_model,
    '3layer': get_3layer_model
}
