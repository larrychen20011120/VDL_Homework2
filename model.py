from torchvision.models.detection import (
    faster_rcnn,
    fasterrcnn_resnet50_fpn_v2, 
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)

def get_model(num_classes=11):

    ### ResNet ###
    # load the pretrained weights -> V2 is better
    # weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # model = fasterrcnn_resnet50_fpn_v2(
    #     weights=weights,
    #     trainable_backbone_layers=2,
    # )

    ### MobileNetV3 ###
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        trainable_backbone_layers=6,
    )
    
    # get the input features of the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # construct the new prediction head
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == "__main__":
    model = get_model()
    print(f"The # of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")