from src.models.efficientnet_cnn.model import get_model as get_effnet
from src.models.vit.model import get_model as get_vit
from src.models.denoised_cnn.model import get_model as get_denoised
from src.models.dct_cnn.model import get_model as get_dct

models = {
    'efficientnet_cnn': get_effnet(pretrained=False),
    'vit': get_vit(pretrained=False),
    'denoised_cnn': get_denoised(pretrained=False),
    'dct_cnn': get_dct(pretrained=False)
} 

import numpy as np
import torch

dummy_input = torch.from_numpy(np.random.rand(2, 3, 224, 224).astype(np.float32))

for name, model in models.items():
    try:
        out = model(dummy_input)
        print(f"{name} works! Output shape: {out.shape}")
        assert out.shape == (2,) or out.shape == (2, 1)
    except Exception as e:
        print(f"{name} failed: {e}")
