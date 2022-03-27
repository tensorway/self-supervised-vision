#%%
import timm
import torch as th
import torch
from torch import nn
from mini_resnet import get_mini_resnet
import torch.nn.functional as F


class BenchmarkModel(nn.Module):
    def __init__(
        self, 
        projector_mlp_arch,
        model_name='mini_resnet20',
        n_classes = 10,
    ):
        '''
        simple embedding based model used as a benchmark

        Args:
            - projector_mlp_arch: List[int] 
                architecture of the mlp that will be appended to the backbone 
                does not include the backbone last layer
            - model_name: string
                the name of the pretrained model
        '''
        super().__init__()
        th.hub._validate_not_a_forked_repo=lambda a,b,c: True
        if 'mini_resnet' in model_name:
            n_layers = int(model_name.split('mini_resnet')[-1])
            self.backbone = get_mini_resnet(n_layers)
            nfeatures = 64
        elif model_name == 'mobilenet_v2':
            self.backbone = th.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
            nfeatures = 1280
            self.backbone.classifier[1] = nn.Identity()
        elif 'efficientnetv2' in model_name:
            self.backbone = timm.create_model(model_name, pretrained=True)
            nfeatures = self.backbone.classifier.weight.shape[1]
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in model_name:   
            self.backbone = th.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
            nfeatures = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        self.projector = MLP(net_arch=[nfeatures]+projector_mlp_arch)
        self.classifier = nn.Linear(nfeatures, n_classes)

    def embed(self, x):
        x = self.backbone(x)
        return self.projector(x)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return th.softmax(x, dim=-1)

class MLP(nn.Module):
    '''
    accepts layer sizes and creates a MLP model out of it
    Args:
        net_arch: list of integers denoting layer sizes (input, hidden0, hidden1, ... hiddenn, out)
    '''
    def __init__(self, net_arch, last_activation= lambda x:x):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
        self.last_activation = last_activation
    def forward(self, x):
        h = x
        for lay in self.layers[:-1]:
            h = F.relu(lay(h))
        h = self.layers[-1](h)
        return h

# %%
if __name__ == '__main__':
    model = BenchmarkModel([1000, 1000], 'mini_resnet20')
    batch = th.rand(3, 3, 32, 32)
    print(model(batch).shape)
    print(model(batch))


# %%
