# inference.py

import argparse
import json
import os

import torch
from torchvision import transforms
from PIL import Image
import nibabel as nib
import pydicom

from config import Config
from models import MultiModalNet

def load_img(path: str, mod: str) -> torch.Tensor:
    """
    Load image/volume for given modality, return torch.Tensor ready for model.
    - 2D: JPEG/PNG → PIL → Resize → ToTensor()
    - DICOM: pydicom → pixel_array → PIL → Resize → ToTensor()
    - 3D: NIfTI (.nii/.nii.gz) → nibabel → volume → Tensor
    """
    
    if mod in ['ct', 'mri']:
        # expect NIfTI files for 3D or raw numpy .nii
        vol = nib.load(path).get_fdata().astype('float32')
        # scale to [0,1]
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        tensor = torch.from_numpy(vol).unsqueeze(0)  # [1, D, H, W]
        # Optionally resize 3D – here we assume correct shape
        return tensor

    # 2D or ultrasound/fundus
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.dcm']:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype('float32')
        img = Image.fromarray(arr)
    else:
        img = Image.open(path).convert('RGB')

    tf = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE_2D),
        transforms.ToTensor(),
    ])
    return tf(img)

def main():
    parser = argparse.ArgumentParser(description='Multimodal Health AI inference')
    parser.add_argument('--model', required=True, help='Path to model .pth file')
    for m in Config.MODALITIES:
        parser.add_argument(f'--{m}', help=f'Filepath for {m}')
    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = MultiModalNet().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Prepare inputs
    inputs = {}
    for m in Config.MODALITIES:
        path = getattr(args, m)
        if path:
            tensor = load_img(path, m).to(device)
            if m in ['ct', 'mri']:
                tensor = tensor.unsqueeze(0)  # add batch dim [1,1,D,H,W]
            else:
                tensor = tensor.unsqueeze(0)  # [1,C,H,W]
            inputs[m] = tensor

    if not inputs:
        print(json.dumps({'error': 'No input modality provided'}))
        return

    # Inference
    with torch.no_grad():
        out = model(inputs)       # [1, num_classes]
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    # Build result dict
    result = dict(zip(Config.DISEASES, [float(p) for p in probs]))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
