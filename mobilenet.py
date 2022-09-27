import torch
import torch.nn as nn
import torchvision.models as models

class MobileNet(nn.Module):
    def __init__(self, dataset, pretrained=True):
		    super(MobileNet, self).__init__()
		    num_classes = 6
		    self.model = models.mobilenet_v3_small(pretrained=pretrained)
		    self.model.fc = nn.Linear(2048, num_classes)
    def forward(self,x):
        output = self.model(x)
        return output  
