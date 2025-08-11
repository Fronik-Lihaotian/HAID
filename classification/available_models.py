import torch.nn as nn
from torchvision import models


def load_model(model_name, pretrained=False, num_classes=100):
    """
    load the model in this method
    :param model_name: choose the model within [resnet50, vit, mv2]
    :param pretrained: Ture for using pretrained model, default False
    :param num_classes: number of classes
    """
    if model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            print('pretrained {} loaded'.format(model_name))
            # TODO finish the pretrained part
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            for name, param in model.named_parameters():
                if not name.split('.')[0] == 'fc':
                    param.requires_grad_(False)
            print('pretrained {} param frozen'.format(model_name))
        else:
            model = models.resnet50(num_classes=num_classes)
            print('Training {} from scratch'.format(model_name))
    elif model_name == 'vit':
        if pretrained:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            print('pretrained {} loaded'.format(model_name))
            # TODO finish the pretrained part
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            for name, param in model.named_parameters():
                if not name.split('.')[0] == 'heads':
                    param.requires_grad_(False)
            print('pretrained {} param frozen'.format(model_name))
        else:
            model = models.vit_b_16(num_classes=num_classes)
            print('Training {} from scratch'.format(model_name))
    elif model_name == 'mobilenet_v2' or model_name == 'mv2':
        if pretrained:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            print('pretrained {} loaded'.format(model_name))
            # TODO finish the pretrained part
            model.classifier[-1] = nn.Linear(model.last_channel, num_classes)
            for param in model.features.parameters():
                param.requires_grad_(False)
            print('pretrained {} param frozen'.format(model_name))
        else:
            model = models.mobilenet_v2(num_classes=num_classes)
            print('Training {} from scratch'.format(model_name))
    else:
        raise ValueError('no such model option, change the model name')

    return model





