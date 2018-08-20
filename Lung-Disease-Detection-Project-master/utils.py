import torch
from torch.autograd import Variable
from torchvision import transforms
from create_dict import *
from models import *
from PIL import Image
import torch.nn.functional as F
import numpy as np

def to_var(x, volatile=False):

    if torch.cuda.is_available():
        x = x.cuda()

    return Variable(x, volatile=volatile)

def infer_from_img_report(cnn_model, rnn_model, image, report, lang_word,val_prob=None):
    
    if torch.cuda.is_available():
        rnn_model.cuda()
        cnn_model.cuda()
    
    rnn_model.eval()
    cnn_model.eval()
    
    report   = torch.from_numpy(sentence2index(lang_word, filter_sen(report))).long()
    report   = to_var(report).unsqueeze_(1).unsqueeze_(2)
    
    seq_lens = torch.LongTensor([len(report)])
    out      = rnn_model(report, seq_lens)
    
    abnormal   = out.cpu().data.numpy().tolist()[0][0]
    
    rnn_out  = np.array([1-abnormal, abnormal])
#     cnn_probs = val_prob
    if val_prob is None:
        cnn_probs = cnn_model(to_var(image.unsqueeze_(0)))
        cnn_out  = F.softmax(cnn_probs, dim=1)[0]
    else:
#         print ('shape of val prob is ',val_prob)
        cnn_out  = F.softmax(to_var(val_prob))
    out =  cnn_out.cpu().data.numpy() + rnn_out
    combined_estim = 0 if out[0]>out[1] else 1
    cnn_out   = cnn_out.cpu().data.numpy()
    rnn_estim =  1 if abnormal>0.5 else 0
    cnn_estim =  0 if cnn_out[0]>cnn_out[1] else 1
    
    return (rnn_estim, cnn_estim, combined_estim), (rnn_out, out/sum(out))
       
if __name__=='__main__':
    name            = 'train'
    train_table     = pd.read_pickle(os.path.join('data','{}_table.pkl'.format(name)))
    train_reports   = train_table.report
    train_reports   = [filter_sen(report) for report in train_reports]
    
    name          = 'val'
    val_table     = pd.read_pickle(os.path.join('data','{}_table.pkl'.format(name)))
    val_reports   = val_table.report
    val_reports   = [filter_sen(report) for report in val_reports]
    lang_word, lang_char = buildDict([val_reports, train_reports])
    
    vocab_size = len(lang_word.word2index)
    #print('vocab: ', vocab_size)
    rnn_model = ReportAnalysis(vocab_size=vocab_size,
                           embedding_dim=256, 
                           hidden_dim=64, 
                           batch_size=1, 
                           num_layers=1,
                           label_size=1,
                           bidirectional=False)

    rnn_model.load_state_dict(torch.load('models/report.pkl'))
    cnn_model = models.resnet18(pretrained=True)
    num_ftrs  = cnn_model.fc.in_features
    cnn_model.fc = nn.Linear(num_ftrs, 2)

    #cnn_model = DenseNet121()
    cnn_model.load_state_dict(torch.load('model.pt'))

    report = "None. Chest pain. Positive TB test. The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax. Normal chest x-XXXX."
    
    image = Image.open('data/images/CXR1_1_IM-0001-3001.png').convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image    = transform(image)
    print(infer_from_img_report(cnn_model, rnn_model, image, report, lang_word))
    