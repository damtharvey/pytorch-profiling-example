from typing import List

import torch.nn as nn
import torchvision.models

MODELS = ['resnet18', 'squeezenet10', 'mobilenet2']


class Ensemble(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.ensemble_size = len(models)
        self.ensemble = nn.ModuleList(models)

    def forward(self, x):
        return sum(model(x) for model in self.ensemble) / self.ensemble_size


class HomogeneousEnsemble(Ensemble):
    def __init__(self, individual: type(nn.Module), ensemble_size: int = 1, **kwargs):
        super().__init__([individual(**kwargs) for _ in range(ensemble_size)])


def homogeneous_ensemble(architecture: str, ensemble_size: int, **kwargs) -> nn.Module:
    if architecture == 'resnet18':
        model = HomogeneousEnsemble(torchvision.models.resnet18, ensemble_size=ensemble_size, **kwargs)
    elif architecture == 'squeezenet10':
        model = HomogeneousEnsemble(torchvision.models.squeezenet1_0, ensemble_size=ensemble_size, **kwargs)
    elif architecture == 'mobilenet2':
        model = HomogeneousEnsemble(torchvision.models.mobilenet_v2, ensemble_size=ensemble_size, **kwargs)
    else:
        raise NotImplementedError("Don't have other architectures.")
    return model
