'''
for debug
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import csv

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

def default_loader(img_path,csv_path):
    img_tensor_list = []
    target_tensor_list = []
    csv_reader = csv.reader(open(csv_path,'r'))
    for num, row in enumerate(csv_reader):
        for i in range(len(row) - 1):
            if i == 0:
                img_pil = Image.open(img_path + row[i])
                img_pil = img_pil.resize((224, 224))
                img_tensor = preprocess(img_pil)
                img_tensor = img_tensor.expand(1, 3, 224, 224)
                bag_tensor = img_tensor
                label = []
                label.append(int(row[-1]))
            else:
                img_pil = Image.open(img_path + row[i])
                img_pil = img_pil.resize((224, 224))
                img_tensor = preprocess(img_pil)
                img_tensor = img_tensor.expand(1, 3, 224, 224)
                bag_tensor = torch.cat((bag_tensor,img_tensor),0)                
        
        bag_tensor_label = torch.LongTensor(label)
        img_tensor_list.append(bag_tensor)
        target_tensor_list.append(bag_tensor_label)
    print(img_tensor_list[0].size()) # [15,3,224,224]
    print(target_tensor_list[0].size()) # [1]
    return img_tensor_list, target_tensor_list

class ThyroidDataset(Dataset):
    def __init__(self, loader=default_loader, img_path=None, csv_path=None):
        self.img_path = img_path
        self.csv_path = csv_path
        csv_reader = csv.reader(open(csv_path,'r'))
        self.instance_list = list(csv_reader)
        self.length = len(self.instance_list)

    def __getitem__(self, index):

        row = self.instance_list[index]
        for i in range(len(row) - 1):
            if i == 0:
                img_pil = Image.open(self.img_path + row[i])
                img_pil = img_pil.resize((224, 224))
                img_tensor = preprocess(img_pil)
                img_tensor = img_tensor.expand(1, 3, 224, 224)
                bag_tensor = img_tensor
                label = []
                label.append(int(row[-1]))
            else:
                img_pil = Image.open(self.img_path + row[i])
                img_pil = img_pil.resize((224, 224))
                img_tensor = preprocess(img_pil)
                img_tensor = img_tensor.expand(1, 3, 224, 224)
                bag_tensor = torch.cat((bag_tensor,img_tensor),0)                

        bag_tensor_label = torch.LongTensor(label)
        return bag_tensor, bag_tensor_label

    def __len__(self):
#         return len(self.img_tensor_list)
        return self.length
