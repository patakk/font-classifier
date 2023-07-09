import torch
import numpy as np
import cv2
import json
import argparse
import glob
from PIL import Image
from torchvision import transforms
from typing import Tuple
from linformer import Linformer
from PIL import Image
from tqdm.notebook import tqdm
from torchvision.models import resnet18

from vit_pytorch.efficient import ViT
from main import ResizeFixedHeight, CustomPadToMinSize
from main import MyDataset

device = 'cuda'

def preprocess_image(cv2_image: np.ndarray, transforms):
    pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    return transforms(pil_image)

def predict(model: torch.nn.Module, cv2_image: np.ndarray, transforms):
    model.eval()
    with torch.no_grad():
        preprocessed_image = preprocess_image(cv2_image, transforms)
        input_tensor = preprocessed_image.unsqueeze(0).to(device)  # Add batch dimension and move to device
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
    return predicted_class

fonts = list(sorted('antonio_y CorporateS-Light.otf akzidenz_grotesk arial calibri din franklin_gothic frutiger helvetica helvetica_bold open_sans_medium open_sans_bold lucida_sans ocrb tahoma verdana monaco_regular officina'.split(' ')))


fonts = [f.split('/')[-1].split('.')[0] for f in glob.glob('ttfs2/*.ttf')]
NUM_CLASSES = len(fonts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict class from image using trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image to be classified')
    parser.add_argument('model_path', type=str, help='Path to the saved model')
    args = parser.parse_args()

    # Load the model
    # efficient_transformer = Linformer(
    #         dim=128,
    #         seq_len=49+1,  # 7x7 patches + 1 cls-token
    #         depth=12,
    #         heads=8,
    #         k=64
    #     )
    # model = ViT(
    #     dim=128,
    #     image_size=224,
    #     patch_size=32,
    #     num_classes=18,
    #     transformer=efficient_transformer,
    #     channels=3,
    # ).to(device)
    model = resnet18(num_classes=NUM_CLASSES).to(device)

    # Load the saved model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Read an image using cv2
    image_path = args.image_path
    gt = json.load(open(image_path.replace('.png', '.json')))['font']
    print('gt', gt)
    cv2_image = cv2.imread(image_path)
    h, w = cv2_image.shape[:2]
    if w > h:
        cv2_image = cv2.resize(cv2_image, (int(224*w/h), 224), cv2.INTER_LINEAR)
    else:
        cv2_image = cv2.resize(cv2_image, (224, int(224*h/w)), cv2.INTER_LINEAR)
    h, w = cv2_image.shape[:2]
    num = int(np.floor(w/h))
    crops = []
    for k in range(num):
        crops.append(cv2_image[:, k*h:(k+1)*h, :])
    
    predictions = []
    for crop in crops:
        # Transforms
        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        # Make a prediction
        prediction = predict(model, crop, image_transforms)
        predictions.append(prediction)
    
    counts = np.bincount(predictions)
    most_common = np.argmax(counts)
    print(gt, '/', fonts[most_common])
    # most frequent font


    #print(f'Predicted class: {fonts[prediction]}')
