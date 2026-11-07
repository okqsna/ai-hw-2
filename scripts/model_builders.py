from torchvision.models.detection import (ssd300_vgg16, SSD300_VGG16_Weights)
from torchvision.models.detection.ssd import SSDClassificationHead

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from transformers import RTDetrForObjectDetection

def build_ssd_model(iou_threshold=0.5, positive_fraction=0.25):
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT,
                         iou_thresh=iou_threshold, 
                         positive_fraction=positive_fraction)
    in_channels = [layer.in_channels for layer in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels=in_channels,
                                                           num_anchors=num_anchors,
                                                           num_classes=9)
    return model

def build_frcnn_model(roi_positive_fraction=0.25):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_positive_fraction=roi_positive_fraction)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=9)

    return model

def build_rtdetr(CHECKPOINT, id_to_label, label_to_id, **params):
    return RTDetrForObjectDetection.from_pretrained(CHECKPOINT,
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        **params)