import torch, torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import AnchorGenerator, FastRCNNPredictor, FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import mobilenet_v2, resnet101, vgg19

# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def get_model(num_classes, backbone = None):
    '''
    Model function to output network according to arguments.
    Args:
        num_classes: number of classes(total_classes+1 for background)
        backbone: to design network with other backbone, default backbone
                  of faster RCNN is resnet50.
    Returns:
        model.
    '''
    
    if backbone == 'mobile_net': 
        net = mobilenet_v2(pretrained = True)
        backbone_ft = net.features
        backbone_ft.out_channels = 1280
        
    elif backbone == 'vgg19':
        net = vgg19(pretrained = True)
        backbone_ft = net.features
        backbone_ft.out_channels = 512 
    
    elif backbone == 'resnet101':
        net = resnet101(pretrained = True)
        modules = list(net.children())[:-1]
        backbone_ft = nn.Sequential(*modules)
        backbone_ft.out_channels = 2048
        
    if backbone is None:
        
        model = fasterrcnn_resnet50_fpn(pretrained = True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # print(in_features) = 1024
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                          num_classes)
        return model
    
    else:

        anchor_gen = AnchorGenerator(sizes=((32, 64, 128),))
        # featmap_names = [0] gives list index out of range error.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names = ['0'],
                                                        output_size = 7,
                                                        sampling_ratio = 2)
        model = FasterRCNN(backbone_ft,
                           num_classes,
                           rpn_anchor_generator = anchor_gen,
                           box_roi_pool = roi_pooler)
        
        return model
