from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
import ast

# TRAIN_MODE set to True if model already trained, False for inference
TRAIN_MODE = True

# Dataset Location
IMG_DIR_PATH = '../input/yandextoloka-water-meters-dataset/WaterMeters/images'
ANNOTATIONS_CSV_PATH = '../input/yandextoloka-water-meters-dataset/WaterMeters/data.csv'
MODEL_PATH = '' #### TODO: put the right path when TRAIN_MODE = False

#Resize SIZE
SIZE = 400

# PYTORCH parameters
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 8
NB_EPOCHS = 8
NB_ITERATION = 1

#first create a dictionnary linking index to image name
# id values goes from 1 to 1244, I reframed that to have id from 0 to 1243.

INT_TO_NAME = {}
for dirname, _, filenames in os.walk(IMG_DIR_PATH):
    for filename in filenames:
        split = filename.split("_")
        INT_TO_NAME[int(split[1])-1] = filename
        
print('example: indice 2 file name -', INT_TO_NAME[1])
print('example: indice 1 file name -', INT_TO_NAME[0])


location_test = pd.read_csv('../input/yandextoloka-water-meters-dataset/WaterMeters/data.csv').iloc[108,2]

def location_to_bounding_boxe(location, size):
    coordinate_list = ast.literal_eval(location)['data']
    x_s = [pt['x']*size for pt in coordinate_list]
    y_s = [pt['y']*size for pt in coordinate_list]
    return max(min(x_s),0), min(min(y_s),size),  max(max(x_s),0), min(size,max(y_s))
    
print("the location:", location_test)
print("---")
print("the corresponding bounding boxe",location_to_bounding_boxe(location_test, 460))


transform_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize([SIZE,SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#ResNet Normalization
    ])

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None):
        self.img_labels = pd.read_csv(annotations_file).set_index('photo_name')
        self.img_dir = img_dir
        
        self.size = SIZE
        
        if transform:
            self.transform = transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        name = INT_TO_NAME[idx]
        img_path = os.path.join(self.img_dir, name)
        image = read_image(img_path)
        reading = self.img_labels.loc[name, 'value']
        bounding_boxe =  location_to_bounding_boxe(self.img_labels.loc[name, 'location'], self.size)
        
        if self.transform:
            image = self.transform(image)
        
        return image, reading, torch.as_tensor(bounding_boxe, dtype=torch.float32)
      
      
waterMeterDataset = ImageDataset(ANNOTATIONS_CSV_PATH, IMG_DIR_PATH, transform = transform_)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
        
 from torchvision.utils import draw_bounding_boxes
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["figure.figsize"] = (20,80)

loader = DataLoader(waterMeterDataset, batch_size=8, shuffle=True)
img, _, bounding_boxes  = next(iter(loader))
img = (img.cpu()*255).type(torch.uint8)
images = [
    draw_bounding_boxes(image, boxes=boxe.unsqueeze(0), width=4)
    for image, boxe in zip(img, bounding_boxes)
]
show(images)


figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(waterMeterDataset), size=(1,)).item()
    img, reading, bounding_boxe = waterMeterDataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(reading)
    plt.axis("off")
    plt.imshow(img.squeeze().transpose(0,1).transpose(1,2), cmap="gray")
plt.show()


train_size = int(0.9*len(waterMeterDataset))
