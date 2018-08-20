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
from models import *
from PIL import Image
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import *
from create_dict import *
from torch.utils.data.dataloader import _use_shared_memory
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class XrayDataset(data.Dataset):
	""" Xray dataset """

	def __init__(self, data, labels, transform=None):
		super(XrayDataset, self).__init__()
	
		self.data   = data
		self.labels = labels
		self.transform = transform
	def __getitem__(self, index):
		"""
		returns 
		"""
				
		image  = torch.from_numpy(self.data[index]).transpose(0,2).float()
		target = torch.FloatTensor([int(self.labels[index])])  
		    
		if self.transform is not None:
			image = self.transform(image)

		return image, target

	def __len__(self):
		"""length of dataset"""
		return self.data.shape[0]


class OpenIDataset(data.Dataset):

	def __init__(self, data_dir='data', name = 'train', transform=None):
		self.table     = pd.read_pickle(os.path.join(data_dir,'{}_table.pkl'.format(name)))
		self.data_dir  = data_dir
		self.transform = transform

	def __len__(self):
		return len(self.table)
		
	def __getitem__(self, idx):
		img_name = self.table.iloc[idx, 0]
		fullname = os.path.join(self.data_dir, img_name+'.png')
		image  = Image.open(fullname).convert('RGB')
		if self.transform:
			image = self.transform(image)
			
		label  = int(self.table.iloc[idx, 2])
		
		return image, label
		
class OpenIReportDataset(data.Dataset):

	def __init__(self, lang, data_dir='data', name = 'train'):

		self.table     = pd.read_pickle(os.path.join('data',
										'{}_table.pkl'.format(name)))
		self.data_dir  = data_dir
		self.lang      = lang
		
	def __len__(self):
		return len(self.table)
		
	def __getitem__(self, idx):

		report = self.table.iloc[idx, 1]
		#print(filter_sen(report))
		report = torch.from_numpy(sentence2index(self.lang, filter_sen(report))).long()	
		label  = int(self.table.iloc[idx, 2])
		
		return report, label

def report_collate(batch):
	
	#print('batch info : ', batch[0][0].shape)
	
	batch.sort(key=lambda x: x[0].shape[0], reverse=True)
	reports, targets = zip(*batch)

	N = len(batch) 
		
	report_lens    = torch.LongTensor([report.shape[0] for report in reports])
	max_report_len = max(report_lens)
	
	if _use_shared_memory:
		data = torch.LongStorage._new_shared(max_report_len, N).new(max_report_len, N).zero_()
	else:
		data = torch.LongTensor(max_report_len, N).zero_()
	labels = torch.LongTensor(list(targets))
	
	for i, report in enumerate(reports):
		data[:report.shape[0], i]  = report
		
	dict_ = {
			'reports':data.unsqueeze(2),
			'seq_lens':report_lens,
			'labels':labels
			}
	return dict_
	



		
if __name__ == '__main__':

	data_dir  = 'data/images'
	name      = 'train'
	train_table     = pd.read_pickle(os.path.join(data_dir,'{}_table.pkl'.format(name)))	
	train_reports   = train_table.report
	train_reports   = [filter_sen(report) for report in train_reports]
	name          = 'val'
	val_table     = pd.read_pickle(os.path.join(data_dir,'{}_table.pkl'.format(name)))	
	val_reports   = val_table.report
	val_reports   = [filter_sen(report) for report in val_reports]


	lang_word, lang_char = buildDict([val_reports, train_reports])

	batch_size  = 8  
	val_ds      = OpenIReportDataset(lang_word, data_dir, name='train')
	data_loader = data.DataLoader(val_ds, 
						 batch_size = batch_size,
						 shuffle = True,
						 collate_fn = report_collate)
						 
	vocab_size, embedding_dim, hidden_dim, num_layers = (len(lang_word.word2index), 
														 128, 
														 128,
														 3)
	 
	embeddings = nn.Embedding(vocab_size+1, embedding_dim)
	lstm       = nn.LSTM(input_size=embedding_dim, 
							  hidden_size=hidden_dim,
							  num_layers = num_layers, 
							  bidirectional=True)
	linear     = nn.Linear(2*hidden_dim, 2)
	model = ReportAnalysis(vocab_size,
						   embedding_dim, 
						   hidden_dim, 
						   batch_size, 
						   num_layers=3,
						   label_size=1)
	print(model)
	for batch_index, batch_dict in enumerate(data_loader):
		print('reports: ', batch_dict['reports'].shape)
		print('labels: ', batch_dict['labels'])
		print('seq_lens: ', batch_dict['seq_lens'])

		reports  = to_var(batch_dict['reports'])
		seq_lens = batch_dict['seq_lens']
		out = model(reports, seq_lens)
		
		print('model out: ', out)		
		
		break

	'''
	data_dir    = 'data/images'

	img_size    = 224 

	train_trans = transforms.Compose([
	transforms.RandomResizedCrop(img_size),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	test_trans = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	val_ds     = OpenIDataset(data_dir, transform=train_trans, name='train')
	data_loader = data.DataLoader(val_ds, 
						 batch_size = 8,
						 shuffle = True)
						 

	for batch_index, (images, labels) in enumerate(data_loader):
	print('images: ', images.shape)
	print('labels: ', labels)
	break


	images = np.load('data/images.npy')
	labels = np.load('data/targets.npy')

	xrayds = XrayDataset(images, labels)
	batch_size = 8
	data_loader = data.DataLoader(
			  xrayds, batch_size = batch_size,
			  sampler = RandomSampler(xrayds))
	model = ResNetCNN(1)
	if torch.cuda.is_available():
	model.cuda()

	criterion = nn.BCELoss()
	params    = list(list(model.linear1.parameters()) + list(model.linear2.parameters()))
	optimizer = torch.optim.Adam(params, lr=1e-4)

	for batch_index, (images, labels) in enumerate(data_loader):
	images, labels = to_var(images), to_var(labels)
	model.zero_grad()
	out  = model(images)
	loss = criterion(out, labels)
	loss.backward()
	optimizer.step()
	print(loss)
	'''
