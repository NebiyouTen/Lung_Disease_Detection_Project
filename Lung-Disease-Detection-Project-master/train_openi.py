import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from models import * 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from data_loader import *

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
def validate(model, data_loader, criterion):
    

	model.eval()
	correct  = 0
	val_loss = 0
		
	for i, (images, labels) in enumerate(data_loader):

		images, labels = to_var(images), to_var(labels)
		out  = model(images)

		pred = out.data.max(1, keepdim=True)[1].int()
		predicted = pred.eq(labels.data.view_as(pred).int())
		correct += predicted.sum()

		loss = criterion(out, labels)
		val_loss += loss.data.sum()
	print('val acc : ', correct/770)
	return val_loss

def main(args):

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

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

	train_ds   = OpenIDataset(data_dir, transform=train_trans, name='train')
	train_data_loader = data.DataLoader(train_ds, 
							 batch_size = args.batch_size,
							 shuffle = True)
							 
	val_ds     = OpenIDataset(data_dir, transform=train_trans, name='val')
	val_data_loader = data.DataLoader(val_ds, 
							 batch_size = args.batch_size,
							 shuffle = True)

	model = DenseNet121()
	criterion = nn.CrossEntropyLoss()


	
	try:
		model.load_state_dict(torch.load(args.resnet_path))
		print("using pre-trained model")
	except:
		print("using new model")

	if torch.cuda.is_available():
		model.cuda()
		criterion.cuda()
	# Loss and Optimizer
	params     = list(
					  list(model.linear1.parameters()) 
					+ list(model.linear2.parameters())
					+ list(model.linear3.parameters())
					)
	
	optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
	
	print('validating.....')
	n_train_batchs   = len(train_ds)//args.batch_size
	n_val_batchs     = len(val_ds)//args.batch_size
	
	best_val = validate(model, val_data_loader, criterion)/n_val_batchs
	
	print("starting val loss {:f}".format(best_val))
	
	for epoch in range(args.num_epochs):

		model.train()
		for i, (images, labels) in enumerate(train_data_loader):
			images, labels = to_var(images), to_var(labels)

			# Forward, Backward and Optimize
			optimizer.zero_grad()
			out  = model(images)
			loss = criterion(out, labels)
			loss.backward()
			optimizer.step()

			# Print log info
			if i % args.log_step == 0:
			    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
			          %(epoch, args.num_epochs, i, n_train_batchs, 
			            loss.data[0])) 
			
			# Save the models
		if (epoch+1) % args.save_step == 0:
				val_loss = validate(model, val_data_loader, criterion)/n_val_batchs
				print('val loss: ', val_loss)
				if val_loss < best_val:
					best_val = val_loss
					print("Found new best val")
					torch.save(model.state_dict(), 
						   os.path.join(
						   				args.model_path, 
						   				args.resnet_path, 
										))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' ,
                        help='path for saving trained models')
    parser.add_argument('--data_dir', type=str, default='data/' ,
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=1,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1,
                        help='step size for saving trained models')
    parser.add_argument('--resnet_path', type=str, default='resnet152.pkl',
                        help='path for trained encoder')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                       help='number of layers in lstm')  
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--split', type=float, default=0.9)
    
    args = parser.parse_args()
    main(args)
