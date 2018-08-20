import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class ResNetCNN(nn.Module):

	def __init__(self):
		"""Load the pretrained ResNet-152 and replace top fc layer."""
		super(ResNetCNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-1]      # delete the last fc layer.
		self.resnet  = nn.Sequential(*modules)
		self.linear1 = nn.Linear(resnet.fc.in_features, 1024)
		self.linear2 = nn.Linear(1024, 1024)    
		self.linear3 = nn.Linear(1024, 2)
		
		self.init_weights()
		
	def init_weights(self):
		"""Initialize the weights."""
		self.linear1.weight.data.normal_(0.0, 0.02)
		self.linear1.bias.data.fill_(0)

		self.linear2.weight.data.normal_(0.0, 0.02)
		self.linear2.bias.data.fill_(0)

		prop_abnormal = 0.635
		data_dist = torch.FloatTensor([1-prop_abnormal, prop_abnormal])

		self.linear3.weight.data.normal_(0.0, 0.02)
		self.linear3.bias.data = data_dist #fill_(0)

		
	def forward(self, images):
		"""Extract the image feature vectors."""
		features = self.resnet(images)
		print('feats: ', features.shape)
		features = Variable(features.data)
		features = features.view(features.size(0), -1)
		features = F.relu(self.linear1(features))
		features = F.relu(self.linear2(features))
		features = F.relu(self.linear3(features))

		return features
class VGGNet(nn.Module):

	def __init__(self):
		"""Load the pretrained ResNet-152 and replace top fc layer."""
		super(VGGNet, self).__init__()
		vggnet = models.vgg19(pretrained=True)
		#print(vggnet)
	
		modules = list(vggnet.children())[:-1]      # delete the last fc layer.
		self.vggnet  = nn.Sequential(*modules)
	
		self.linear1 = nn.Linear(vggnet.classifier[0].in_features, 1024)
		self.linear2 = nn.Linear(1024, 1024)    
		self.linear3 = nn.Linear(1024, 2)
		self.init_weights()
		
	def init_weights(self):
		"""Initialize the weights."""
		self.linear1.weight.data.normal_(0.0, 0.02)
		self.linear1.bias.data.fill_(0)

		self.linear2.weight.data.normal_(0.0, 0.02)
		self.linear2.bias.data.fill_(0)

		prop_abnormal = 0.635
		data_dist = torch.FloatTensor([1-prop_abnormal, prop_abnormal])

		self.linear3.weight.data.normal_(0.0, 0.02)
		self.linear3.bias.data = data_dist #fill_(0)

		
	def forward(self, images):
		"""Extract the image feature vectors."""
		features = self.vggnet(images)
		features = Variable(features.data)
		features = features.view(features.size(0), -1)
		features = F.relu(self.linear1(features))
		features = F.relu(self.linear2(features))
		features = F.relu(self.linear3(features))

		return features
		
class VGG16(nn.Module):

    def __init__(self):
        """Load the pretrained VGG-16 and replace top fc layer."""
        super(VGG16, self).__init__()
        self.vggnet = models.vgg16(pretrained=True)
        self.vggnet.classifier = nn.Sequential(
                                 nn.Linear(self.vggnet.classifier[0].in_features, 1024),
                                 nn.Linear(1024, 17))
 
            
        #self.init_weights()

    def init_weights(self):
        """Initialize the weights."""

        label_dist = torch.FloatTensor([0.47966632, 0.06534585, 0.08689607, 0.04066736, 0.05943691, 0.06047967,
        0.03267292, 0.04935697, 0.01668405, 0.02294056, 0.0132082,  0.01494612,
        0.01529371, 0.01668405, 0.00729927, 0.00868961, 0.00973236])

        self.vggnet.classifier.weight.data.uniform_(-0.1, 0.1)
        self.vggnet.classifier.bias.data = label_dist

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.vggnet(images)

        return features

class DenseNet121(nn.Module):

	def __init__(self):
		"""Load the pretrained DenseNet-121 and replace top fc layer."""
		super(DenseNet121, self).__init__()
		densenet = models.densenet121(pretrained=True)
		#print(densenet)
		modules  = list(densenet.children())[:-1]      # delete the last fc layer.
		modules.append(nn.AvgPool2d(kernel_size=7))
		self.features   = nn.Sequential(*modules)
		#print(densenet.classifier.in_features)
		self.classifier = nn.Sequential(
						  nn.Linear(densenet.classifier.in_features, 1024),
						  nn.ReLU(),
						  nn.Linear(1024, 1024),
						  nn.ReLU(),    
						  nn.Linear(1024, 2)
						  )
		
		#self.init_weights()
		
	def init_weights(self):
		"""Initialize the weights."""
		self.classifier[0].weight.data.normal_(0.0, 0.02)
		self.classifier[0].bias.data.fill_(0)

		self.classifier[2].weight.data.normal_(0.0, 0.02)
		self.classifier[2].bias.data.fill_(0)

		prop_abnormal = 0.635
		data_dist = torch.FloatTensor([1-prop_abnormal, prop_abnormal])

		self.classifier[4].weight.data.normal_(0.0, 0.02)
		self.classifier[4].bias.data = data_dist #fill_(0)

		
	def forward(self, images):
		"""Extract the image feature vectors."""
		features = self.features(images)
		features = Variable(features.data)
		features = features.view(features.size(0), -1)
		features = self.classifier(features)

		return features

class ReportAnalysis(nn.Module):

	def __init__(self, vocab_size, embedding_dim, hidden_dim,
					   batch_size, num_layers=1, label_size=1,
					   dropout=0.5, bidirectional=False):

		super(ReportAnalysis, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim    = hidden_dim
		self.batch_size    = batch_size
		self.drop = nn.Dropout(dropout)
		self.embeddings = nn.Embedding(vocab_size+1, embedding_dim)
		self.lstm       = nn.LSTM(input_size=embedding_dim, 
								  hidden_size=hidden_dim,
								  num_layers = num_layers,
								  dropout=dropout,
								  bidirectional=bidirectional)	
		self.bimul = 1
		if bidirectional:self.bimul=2				  
		self.projection = nn.Linear(hidden_dim*self.bimul, label_size)
		self.num_layers = num_layers
		#if not bidirectional:
		#	 self.projection.weight = self.embeddings.weight 
		self.init_weights()
		self.hidden = self.init_hidden()

	def init_weights(self):
		"""Initialize the weights."""
		initrange = 0.1
		self.projection.weight.data.uniform_(-initrange, initrange)
		prop_abnormal = 0.635
		data_dist = torch.FloatTensor([prop_abnormal])
		self.projection.bias.data= data_dist#.fill_(0)

	def init_hidden(self):

		if torch.cuda.is_available():
		
		    return (Variable(torch.zeros(self.bimul*self.num_layers, 
		    							 self.batch_size, self.hidden_dim).cuda()),
		            Variable(torch.zeros(self.bimul*self.num_layers, 
		            					 self.batch_size, self.hidden_dim).cuda()))
		else:
		    return (Variable(torch.zeros(self.bimul*self.num_layers,
		    							 self.batch_size, self.hidden_dim)),
					Variable(torch.zeros(self.bimul*self.num_layers, 
										 self.batch_size, self.hidden_dim)))		

	def forward(self, reports, seq_lens):

		emb = self.embeddings(reports.squeeze(2))
		packed_input = pack_padded_sequence(emb, seq_lens.cpu().numpy())
		out,  self.hidden = self.lstm(packed_input, self.hidden)
		out       = out
		out, lens = pad_packed_sequence(out)
		lengths = [l - 1 for l in seq_lens] #extract last cell output
		out = out[lengths, range(len(lengths))]
		out = self.projection(out)
		
		return F.sigmoid(out)

class EncoderVGG(nn.Module):
    def __init__(self, model_path=None):
        super(EncoderVGG, self).__init__()
        if model_path is None:
            vgg = models.vgg16_bn(pretrained=True)
            self._vgg_extractor = nn.Sequential(*(vgg.features[i] for i in range(35)))
        else:
            self._vgg_extractor = torch.load(model_path)
            
    def forward(self, x):
        return self._vgg_extractor(x)

class Decoder(nn.Module):

    def __init__(self, vis_dim, vis_num, embed_dim, hidden_dim, vocab_size, num_layers=1, dropout_ratio=0.5):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.vis_dim = vis_dim
        self.vis_num = vis_num
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.lstm_cell = nn.LSTMCell(embed_dim + vis_dim, hidden_dim, num_layers)
        self.fc_dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # attention
        self.att_vw = nn.Linear(self.vis_dim, self.vis_dim, bias=False)
        self.att_hw = nn.Linear(self.hidden_dim, self.vis_dim, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(vis_num))
        self.att_w = nn.Linear(self.vis_dim, 1, bias=False)

    def _attention_layer(self, features, hiddens):
        """
        :param features:  batch_size  * 196 * 512
        :param hiddens:  batch_size * hidden_dim
        :return:
        """
        att_fea = self.att_vw(features)
        # N-L-D
        att_h = self.att_hw(hiddens).unsqueeze(1)
        # N-1-D
        att_full = nn.ReLU()(att_fea + att_h + self.att_bias.view(1, -1, 1))
        att_out = self.att_w(att_full).squeeze(2)
        #print(att_out.shape)
        alpha = nn.Softmax()(att_out)
        #print('alpha: ', alpha.shape)
        # N-L
        context = torch.sum(features * alpha.unsqueeze(2), 1)
        return context, alpha

    def forward(self, features, captions, lengths):
        """
        :param features: batch_size * 196 * 512
        :param captions: batch_size * time_steps
        :param lengths:
        :return:
        """
        batch_size, time_step = captions.data.shape
        vocab_size = self.vocab_size
        embed = self.embed
        dropout = self.dropout
        attention_layer = self._attention_layer
        lstm_cell = self.lstm_cell
        fc_dropout = self.fc_dropout
        fc_out = self.fc_out

        word_embeddings = embed(captions)
        word_embeddings = dropout(word_embeddings) if dropout is not None else word_embeddings
        feas = torch.mean(features, 1)  # batch_size * 512
        h0, c0 = self.get_start_states(batch_size)

        predicts = to_var(torch.zeros(batch_size, time_step, vocab_size))

        for step in range(time_step):
            batch_size = sum(i >= step for i in lengths)
            if step != 0:
                feas, alpha = attention_layer(features[:batch_size, :], h0[:batch_size, :])
            #feas  = torch.zeros_like(feas) 
            words = (word_embeddings[:batch_size, step, :]).squeeze(1)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0[:batch_size, :], c0[:batch_size, :]))
            outputs = fc_out(fc_dropout(h0)) if fc_dropout is not None else fc_out(h0)
            predicts[:batch_size, step, :] = outputs

        return predicts

    def sample(self, feature, max_len=20):
        # greedy sample
        embed = self.embed
        lstm_cell = self.lstm_cell
        fc_out = self.fc_out
        attend = self._attention_layer
        batch_size = feature.size(0)

        sampled_ids = []
        alphas = [0]

        words = embed(to_var(torch.ones(batch_size, 1).long())).squeeze(1)
        h0, c0 = self.get_start_states(batch_size)
        feas = torch.mean(feature, 1) # convert to batch_size*512

        for step in range(max_len):
            if step != 0:
                feas, alpha = attend(feature, h0)
                alphas.append(alpha)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0, c0))
            outputs = fc_out(h0)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.unsqueeze(1))
            words = embed(predicted)

        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze(), alphas

    def get_start_states(self, batch_size):
        hidden_dim = self.hidden_dim
        h0 = to_var(torch.zeros(batch_size, hidden_dim))
        c0 = to_var(torch.zeros(batch_size, hidden_dim))
        return

if __name__=='__main__':
	resnet = ResNetCNN()
	densenet = DenseNet121()
	vggnet   = VGGNet()
	#print(densenet)
	#x_test = torch.randn(4,3,224,224)
	#print(densenet(Variable(x_test)))
	
