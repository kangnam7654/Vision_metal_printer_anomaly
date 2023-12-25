import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model="resnet50", out_dim=512):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights = models.ResNet18_Weights.DEFAULT),
                            "resnet50": models.resnet50(weights = models.ResNet50_Weights.DEFAULT)}

        resnet = self._get_basemodel(base_model)
        num_features = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_features, num_features)
        self.l2 = nn.Linear(num_features, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except Exception:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x