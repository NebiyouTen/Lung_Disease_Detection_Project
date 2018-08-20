import os
import numpy as np # linear algebra
import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torchvision
from torchvision import transforms
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt
#from models import *
from PIL import Image
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
#from utils import *
#from create_dict import *
from torch.utils.data.dataloader import _use_shared_memory
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
from build_vocab import *

class OpenIDataset(data.Dataset):

    def __init__(self, vocab, table_path ='.', 
                              feature_path='data/feat_openi.pth', 
                              name = 'train', transform=None):

        self.table = pd.read_pickle(os.path.join(table_path,
                                        '{}_table.pkl'.format(name)))
        self.feature_path = feature_path
        self.transform = transform
        self._vocab    = vocab 
        self._features = torch.load(feature_path)
        self._build_index()
        self._features = self._features[1]

        
    def __len__(self):
        return len(self.table)
    def _build_index(self):
        imageName_to_feature_idx = {}
        for idx, name in enumerate(self._features[0]):
            imageName_to_feature_idx[name] = idx
        self._imageName_to_feature_idx = imageName_to_feature_idx
        
    def _get_caption_tensor(self, caption):
        vocab = self._vocab
        tokens = caption.lower().split('/')
        target = list()
        target.append(vocab('<start>'))
        target.extend([vocab(word) for word in tokens])
        target.append(vocab('<end>'))
        target = torch.Tensor(target)
        return target

    def __getitem__(self, idx):
        img_name = self.table.iloc[idx, 0]
        fullname = img_name+'.png'
        #print(fullname)
        feature_idx = self._imageName_to_feature_idx[fullname]
        feature     = self._features[feature_idx].view(512,-1)
        caption = self.table.iloc[idx, 3]
        caption = self._get_caption_tensor(caption)

        return feature, caption

    def collate_fn(self, data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        features, captions = zip(*data)

        features = torch.stack(features, 0)
        # batch_size-by-512-196

        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return features, targets, lengths

def get_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    return transform


def get_loader(vocab, feature_path, batch_size=1,
               shuffle=True, num_workers=2, name='train', transform=None):

    openi = OpenIDataset(vocab,
                         feature_path=feature_path,
                         name=name, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=openi,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=openi.collate_fn)
    return data_loader


if __name__=='__main__':
    vocab_path   = 'openi-vocab.pkl'
    feature_path = 'data/feat_openi.pth'
    transform = get_transform()
    name = 'train'
    train_table      = pd.read_pickle(os.path.join('.','{}_table.pkl'.format(name)))	
    train_captions   = train_table.caption
    captions   = [caption for caption in train_captions]
    name          = 'val'
    val_table     = pd.read_pickle(os.path.join('.','{}_table.pkl'.format(name)))	
    val_captions   = val_table.caption
    val_captions   = [caption for caption in val_captions]
    captions.extend(val_captions)
    vocab = build_vocab(captions, threshold=0)



    loader    = get_loader(vocab, feature_path, batch_size=2,
                shuffle=True, num_workers=0, name='train', transform=transform) 
    print('data loading done!!!!!!!')
    for i, (feats, targets, lengths) in enumerate(loader):
        print(feats.shape)
        break	

