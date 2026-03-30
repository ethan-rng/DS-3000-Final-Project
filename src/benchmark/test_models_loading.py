import sys
import os
import torch
print("Python Path:", sys.path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=====================================")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Using Device:   {device}")
print(f"=====================================\n")

try:
    from src.models.vit.model import get_model as get_vit
    from src.models.denoised_cnn.model import get_model as get_denoised
    from src.models.efficientnet_cnn.model import get_model as get_effnet
    from src.models.dct_cnn.model import get_model as get_dct
except ImportError as e:
    print("Import Error:", e)
    sys.exit(1)

print('Instantiating ViT...')
vit = get_vit(pretrained=False).to(device)
print('Instantiating Denoised CNN...')
denoised = get_denoised(pretrained=False).to(device)
print('Instantiating EfficientNet...')
effnet = get_effnet(pretrained=False).to(device)
print('Instantiating DCT CNN...')
dct = get_dct(pretrained=False).to(device)

x = torch.randn(2, 3, 224, 224).to(device)
print('Testing forwards...')
out_vit = vit(x)
assert out_vit.shape == (2,), f'ViT failed: {out_vit.shape}'

out_denoised = denoised(x)
assert out_denoised.shape == (2,), f'Denoised failed: {out_denoised.shape}'

out_effnet = effnet(x)
assert out_effnet.shape == (2,), f'EffNet failed: {out_effnet.shape}'

out_dct = dct(x)
assert out_dct.shape == (2,), f'DCT failed: {out_dct.shape}'

print('All models successfully instantiated and forwarded!')
