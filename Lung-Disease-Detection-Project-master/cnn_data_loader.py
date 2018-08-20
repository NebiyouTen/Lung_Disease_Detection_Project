import os
import numpy as np # linear algebra
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
import pickle

class OpenICNNDataset(data.Dataset):

	def __init__(self, data_dir='data/images', table_path ='data/', 
							  name = 'train', transform=None):
							  
		self.table = pd.read_pickle(os.path.join(table_path,
										'cnn_{}_table.pkl'.format(name)))
		self.transform = transform
		self.data_dir = data_dir
	def __len__(self):
		return len(self.table)

	def __getitem__(self, idx):
		img_name = self.table.iloc[idx, 0]
		fullname = img_name+'.png'
		image = Image.open(os.path.join(self.data_dir , fullname)).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)
		label = self.table.iloc[idx, 1]

		return image, label

if __name__ =='__main__':
	img_size    = 224 

	transform = transforms.Compose([
	transforms.RandomResizedCrop(img_size),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	dataset     = OpenICNNDataset(transform=transform)
	data_loader = torch.utils.data.DataLoader( dataset=dataset, batch_size=8,
		                                   shuffle=True, num_workers=0)

	for batch_index, (images, labels) in enumerate(data_loader):

		print(images.shape)
		break


