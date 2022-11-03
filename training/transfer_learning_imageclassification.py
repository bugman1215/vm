import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

%matplotlib inline
import os
os.getcwd()
# place the files in your IDE working dicrectory .
labels = pd.read_csv(r'/aerialcactus/train.csv')
submission = pd.read_csv(r'/aerialcactus/sample_submission.csv)

train_path = r'/aerialcactus/train/train/'
test_path = r'/aerialcactus/test/test/'
from transformers import SegformerFeatureExtractor, SegformerForImageClassification
from PIL import Image
import requests

                         
labels.head()
labels.tail()                         
labels['has_cactus'].value_counts()
label = 'Has Cactus', 'Hasn\'t Cactus'
plt.figure(figsize = (8,8))
plt.pie(labels.groupby('has_cactus').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()
                         
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)



if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)

