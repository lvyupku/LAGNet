import csv
import os
import tarfile
from urllib.parse import urlparse
from urllib.request import urlretrieve
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import pandas as pd

class Orid(Dataset):
    def __init__(self, root, csv_path, transform=None):
        self.root = root
        self.transform = transform
        self.datas = pd.read_excel(csv_path)
        
        
    def __getitem__(self, index):
        row = self.datas.iloc[[index]]
        label = np.array(row[['D','G', 'C', 'A', 'H', 'M', 'O']])
        label = torch.tensor(label[0])
        #print(label)
        imgpath = os.path.join(self.root, row['Fundus'].values[0])

        img = Image.open(imgpath).convert('RGB')
       
        if self.transform is not None:
            img = self.transform(img)

        data = {'image':img, 'target': label}
        return data

    def __len__(self):
        return len(self.datas)