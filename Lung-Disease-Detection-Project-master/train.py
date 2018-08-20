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
    
    val_size = len(data_loader)
    val_loss = 0
    model.train()
    
    for i, (images, labels) in enumerate(train_data_loader):
    
        images, labels = to_var(images), to_var(labels)
        out = model(images)
        loss = criterion(out, labels)
        val_loss += loss.data.sum()

    return val_loss/val_size
    
def main(args):

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    images = np.load(
                 os.path.join(
                 args.data_dir,
                 'images.npy'),
                encoding='latin1')

    labels = np.load(
             os.path.join(
             args.data_dir,
             'targets.npy'),
              encoding='latin1')

    print('labels shape: ', labels.shape)
    print('img data: ', img_data.shape)

    train_scanpath_ds = XrayDataset(images, labels)

    train_data_loader = data.DataLoader(
                                 train_scanpath_ds, batch_size = args.batch_size,
                                 sampler = RandomSampler(train_scanpath_ds))
    #val_scanpath_ds = ScanpathDataset(val_data, val_labels, vocab)

    #val_data_loader = data.DataLoader(
    #				             val_scanpath_ds, batch_size = args.batch_size,
    #				             sampler = RandomSampler(val_scanpath_ds),
    #				             collate_fn = collate_fn)

    print(len(train_data_loader))
    model = ResNetCNN(1)
    if torch.cuda.is_available():

    criterion = nn.BCELoss()

    try:
        model.load_state_dict(torch.load(args.resnet_path))
        print("using pre-trained model")
    except:
        print("using new model")

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    # Loss and Optimizer
    params     = list(list(model.linear1.parameters()) + list(model.linear2.parameters()))
    optimizer  = torch.optim.Adam(params, lr=1e-4)
    total_step = len(train_data_loader)
    #print('validating.....')
    #best_val = validate(encoder, decoder, val_data_loader, criterion)
    #print("starting val loss {:f}".format(best_val))
    for epoch in range(args.num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_data_loader):
            images, labels = to_var(images), to_var(labels)

            # Forward, Backward and Optimize
            model.zero_grad()

            out = model(images)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0])) 

            # Save the models
        if (epoch+1) % args.save_step == 0:
    #			val_loss = validate(encoder, decoder, val_data_loader, criterion)
    #			print('val loss: ', val_loss)
    #			if val_loss < best_val:
    #				best_val = val_loss
    #			print("Found new best val")
            torch.save(model.state_dict(), 
                       os.path.join(args.resnet_path, 
                                    'resnet152.pkl'))

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
    parser.add_argument('--resnet_path', type=str, default='./models/resnet152.pkl',
                        help='path for trained encoder')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                       help='number of layers in lstm')  
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--split', type=float, default=0.9)
    
    args = parser.parse_args()
    main(args)
