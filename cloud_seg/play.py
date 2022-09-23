import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import Unet
from utils import load_checkpoint
from model import Unet
import torchvision
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


def single_prediction(image_path, model, folder="data/saved_images/", device=DEVICE):
    model.eval()

    pillow_image = Image.open(image_path)
    image = np.array(pillow_image)

    transformed = transform(image=image)
    x = transformed["image"]

    x = torch.unsqueeze(x, dim=0)
    
    x = x.to(device=device)
    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()

    torchvision.utils.save_image(preds, f"{folder}/pred_new.png")

    model.train()
    

    

if __name__ == "__main__":
    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("my_checkpoints.pth.tar"), model)
    single_prediction("data/val/test_image.png", model)