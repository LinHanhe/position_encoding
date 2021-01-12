import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
import torch.nn as nn


class EncodeLoss(nn.Module):
    def __init__(self):
        super(EncodeLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()

    def forward(self, outputs, labels):
        loss = 0
        for idx in range(5):
            loss += torch.mean(self.bce(self.sigmoid(outputs[:,idx]),labels[:,idx]))
            loss += torch.mean(labels[:,idx]*self.bce(self.sigmoid(outputs[:,idx+5]),labels[:,idx+5]))
 
        return loss

class EncodeMetric(nn.Module):
    def __init__(self):
        super(EncodeMetric, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, labels):
        outputs_ = (self.sigmoid(outputs) > 0.5).type(torch.FloatTensor)
        outputs_[:,5:] = outputs_[:,:5]*outputs_[:,5:]
        labels_ = labels.to('cpu')
        num_correct = torch.sum((torch.sum(outputs_-labels_,1)==0).type(torch.FloatTensor))
        
        return num_correct




def helmet_use_encode():
    
    names_to_labels = {}
    names_to_labels['DHelmet'] = [1,0,0,0,0,1,0,0,0,0]
    names_to_labels['DNoHelmet'] =  [1,0,0,0,0,0,0,0,0,0]
    names_to_labels['DHelmetP0Helmet'] = [1,1,0,0,0,1,1,0,0,0]
    names_to_labels['DHelmetP0NoHelmet'] = [1,1,0,0,0,1,0,0,0,0]
    names_to_labels['DNoHelmetP0NoHelmet'] = [1,1,0,0,0,0,0,0,0,0]
    names_to_labels['DHelmetP1Helmet'] = [1,0,1,0,0,1,0,1,0,0]
    names_to_labels['DNoHelmetP1Helmet'] = [1,0,1,0,0,0,0,1,0,0]
    names_to_labels['DHelmetP1NoHelmet'] = [1,0,1,0,0,1,0,0,0,0]
    names_to_labels['DNoHelmetP1NoHelmet'] = [1,0,1,0,0,0,0,0,0,0]
    names_to_labels['DHelmetP0HelmetP1Helmet'] = [1,1,1,0,0,1,1,1,0,0]
    names_to_labels['DHelmetP0NoHelmetP1Helmet'] = [1,1,1,0,0,1,0,1,0,0]
    names_to_labels['DHelmetP0NoHelmetP1NoHelmet'] = [1,1,1,0,0,1,0,0,0,0]
    names_to_labels['DNoHelmetP0NoHelmetP1Helmet'] = [1,1,1,0,0,0,0,1,0,0]
    names_to_labels['DNoHelmetP0NoHelmetP1NoHelmet'] = [1,1,1,0,0,0,0,0,0,0]
    names_to_labels['DNoHelmetP0HelmetP1NoHelmet'] = [1,1,1,0,0,0,1,0,0,0]
    names_to_labels['DHelmetP1HelmetP2Helmet'] = [1,0,1,1,0,1,0,1,1,0]
    names_to_labels['DHelmetP1NoHelmetP2Helmet'] = [1,0,1,1,0,1,0,0,1,0]
    names_to_labels['DHelmetP1HelmetP2NoHelmet'] = [1,0,1,1,0,1,0,1,0,0]
    names_to_labels['DHelmetP1NoHelmetP2NoHelmet'] = [1,0,1,1,0,1,0,0,0,0]
    names_to_labels['DNoHelmetP1HelmetP2Helmet'] = [1,0,1,1,0,0,0,1,1,0]
    names_to_labels['DNoHelmetP1NoHelmetP2Helmet'] = [1,0,1,1,0,0,0,0,1,0]
    names_to_labels['DNoHelmetP1NoHelmetP2NoHelmet'] = [1,0,1,1,0,0,0,0,0,0]
    names_to_labels['DHelmetP0HelmetP1HelmetP2Helmet'] = [1,1,1,1,0,1,1,1,1,0]
    names_to_labels['DHelmetP0HelmetP1NoHelmetP2Helmet'] = [1,1,1,1,0,1,1,0,1,0]
    names_to_labels['DHelmetP0HelmetP1NoHelmetP2NoHelmet'] = [1,1,1,1,0,1,1,0,0,0]
    names_to_labels['DHelmetP0NoHelmetP1HelmetP2Helmet'] = [1,1,1,1,0,1,0,1,1,0]
    names_to_labels['DHelmetP0NoHelmetP1NoHelmetP2Helmet'] = [1,1,1,1,0,1,0,0,1,0]
    names_to_labels['DHelmetP0NoHelmetP1NoHelmetP2NoHelmet'] = [1,1,1,1,0,1,0,0,0,0]
    names_to_labels['DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmet'] = [1,1,1,1,0,0,0,0,0,0]
    names_to_labels['DNoHelmetP0NoHelmetP1NoHelmetP2Helmet'] = [1,1,1,1,0,0,0,0,1,0]
    names_to_labels['DHelmetP1NoHelmetP2NoHelmetP3NoHelmet'] = [1,0,1,1,1,1,0,0,0,0]
    names_to_labels['DHelmetP1NoHelmetP2NoHelmetP3Helmet'] = [1,0,1,1,1,1,0,0,0,1]
    names_to_labels['DNoHelmetP1NoHelmetP2NoHelmetP3NoHelmet'] = [1,0,1,1,1,0,0,0,0,0]
    names_to_labels['DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3NoHelmet'] = [1,1,1,1,1,0,0,0,0,0]
    names_to_labels['DHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3Helmet'] = [1,1,1,1,1,1,0,0,0,1]
    names_to_labels['DHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3NoHelmet'] = [1,1,1,1,1,1,0,0,0,0]
    
    return names_to_labels

class HelmetDataset(Dataset):
    """HELMET dataset."""

    def __init__(self, ids, root_dir, names_to_labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ids = ids
        self.root_dir = root_dir
        self.transform = transform
        self.names_to_labels = names_to_labels
        self.num_class = len(names_to_labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        im_path = self.root_dir+self.ids.iloc[idx, 0]+'/'+"{0:02d}".format(self.ids.iloc[idx, 2])+'.jpg'
        image = Image.open(im_path)
        
        crop_image = image.crop((self.ids.iloc[idx, 3],self.ids.iloc[idx, 4],self.ids.iloc[idx, 3]+self.ids.iloc[idx, 5],self.ids.iloc[idx, 4]+self.ids.iloc[idx, 6]))
        
        label = np.array(self.names_to_labels[self.ids.iloc[idx, 7]])
         
        
        if self.transform:
            img = self.transform(crop_image)

        
        sample = {'image': img, 'label': label}
        
        return sample
