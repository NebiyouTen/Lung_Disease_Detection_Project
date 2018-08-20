import numpy as np
import argparse
from torch.nn.utils.rnn import  pack_padded_sequence
from models import *
from data_loader_openi import *
import torch.optim as optim
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def train(dataloader, model, optimizer, criterion, epoch, total_epoch):
    total_step = len(dataloader)
    # print 'Total step:', total_step
    epoch_loss = 0
    for i, (features, targets, lengths) in enumerate(dataloader):
        optimizer.zero_grad()
        features = to_var(features).transpose(1,2)
        targets = to_var(targets)
        predicts = model(features, targets[:, :-1], [l - 1 for l in lengths])
        predicts = pack_padded_sequence(predicts, [l-1 for l in lengths], batch_first=True)[0]
        targets = pack_padded_sequence(targets[:, 1:], [l-1 for l in lengths], batch_first=True)[0]
        loss = criterion(predicts, targets)
        epoch_loss+=loss.data.sum()
        loss.backward()
        optimizer.step()
        #if (i+1)%30 == 0:
        #    print('Epoch [%d/%d]: [%d/%d], loss: %5.4f, perplexity: %5.4f.'%(epoch, total_epoch,i,
        #                                                                     total_step,epoch_loss/(i+1),
        #                                                                    np.exp(epoch_loss/(i+1))))
    return epoch_loss/total_step

def validate(dataloader, model, criterion):
    total_step = len(dataloader)
    # print 'Total step:', total_step
    model.eval()
    total_loss = 0
    for i, (features, targets, lengths) in enumerate(dataloader):
        features = to_var(features).transpose(1,2)
        targets = to_var(targets)
        predicts = model(features, targets[:, :-1], [l - 1 for l in lengths])
        predicts = pack_padded_sequence(predicts, [l-1 for l in lengths], batch_first=True)[0]
        targets = pack_padded_sequence(targets[:, 1:], [l-1 for l in lengths], batch_first=True)[0]
        loss = criterion(predicts, targets)
        total_loss += loss.data.sum() 
    return total_loss/total_step



def main(args):
    # dataset setting

    feature_path = args.feature_path
    vocab_path = args.vocab_path
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_workers = args.num_workers
    
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
    
    dataloader = get_loader(vocab, 
                            feature_path= feature_path, 
                            batch_size = batch_size,
                            shuffle=True, 
                            num_workers=num_workers, 
                            name='train', 
                            transform=transform) 
    val_dataloader = get_loader(vocab, 
                        feature_path= feature_path, 
                        batch_size = batch_size,
                        shuffle=True, 
                        num_workers=num_workers, 
                        name='val', 
                        transform=transform) 

    
    vocab_size = len(vocab)
   
    # model setting
    vis_dim = args.vis_dim
    vis_num = args.vis_num
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    dropout_ratio = args.dropout_ratio
    
    model = Decoder(vis_dim=vis_dim,
                    vis_num=vis_num, 
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim, 
                    vocab_size=vocab_size, 
                    num_layers=num_layers,
                    dropout_ratio=dropout_ratio)
    
    # optimizer setting
    lr = args.lr
    num_epochs = args.num_epochs
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # criterion
    criterion = nn.CrossEntropyLoss()
    model.cuda()
    model.train()
    
    print('Number of epochs:', num_epochs)
    best_val_loss = validate(val_dataloader, model, criterion)
    print('val_loss: ', best_val_loss)
    
    for epoch in range(num_epochs):
        train_loss = train(dataloader=dataloader, model=model, optimizer=optimizer, criterion=criterion,
              epoch=epoch, total_epoch=num_epochs)
        val_loss = validate(val_dataloader, model, criterion)
        print('='*70)
        print('|Epoch [%d/%d]|validation | %5.2f | %5.2f | Training | %5.2f | %5.2f |'%(epoch, 
                                                                                        num_epochs, 
                                                                                        val_loss,np.exp(val_loss), 
                                                                                        train_loss, np.exp(train_loss)))
        torch.save(model, 'checkpoints/attn_model.pth')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data loader
    parser.add_argument('--feature_path', type=str,
                        default='data/feat_openi.pth')
    parser.add_argument('--vocab_path', type=str,
                        default='openi-vocab.pkl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2)

    # model setting
    parser.add_argument('--vis_dim', type=int, default=512)
    parser.add_argument('--vis_num', type=int, default=196)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=155)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout_ratio', type=float, default=0.8)

    # optimizer setting
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=150)

    args = parser.parse_args()
    main(args)
