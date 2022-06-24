from timm import create_model


def load_feature_extractor(model_name='vgg16', pretrained=True, num_classes=0, device=None):
    if device is None:
        device = 'cpu'

    feature_extractor = (
        create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        .features.eval()
        .to(device)
    )
    return feature_extractor
