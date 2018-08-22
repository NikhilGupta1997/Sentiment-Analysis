import numpy as np
import pickle
import random
import re
import string
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from parse_data import *
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Global Variables
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 256
CLIP = 250

MODE = 'GRU' 	# Cell Type : LSTM or GRU
EPOCHS = 10
LEARNING_RATE = 0.001
SUMMARY_WEIGHT = 1
LAYERS = 1

''' Read Data from file and Parse into useable form'''
def get_data(file):
	count = 0
	all_text = []
	all_ratings = []
	all_summaries = []
	for x in open(file):
		count += 1
		if count % 10000 == 0:
			print("Iteration = {}".format(count))
			sys.stdout.flush()
		data = re.sub('"reviewText":|"overall":|"summary":', '@#^@', x).strip().split("@#^@")
		text = ','.join(data[1].split(',')[:-1]).strip().strip('\"')
		rating = float(data[2].split(',')[0].strip())
		summary = data[3].strip().strip('}').strip('\"')
		all_text.append(parse_text(text))
		all_ratings.append(parse_rating(rating))
		all_summaries.append(parse_summary(summary))
	return all_text, all_ratings, all_summaries

def shuffle_data(data, tags):
	c = list(zip(data, tags))
	random.shuffle(c)
	data, tags = zip(*c)
	return data, tags

def sort_data(data, tags):
	size = len(tags)
	indexes = range(len(data))
	indexes = [x for _, x in sorted(zip(data,indexes), key=lambda pair: len(pair[0]))]
	data = list(map(data.__getitem__, indexes))
	tags = list(map(tags.__getitem__, indexes))
	diff = (BATCH_SIZE - (size % BATCH_SIZE)) % BATCH_SIZE
	data = data + data[-diff:]
	tags = tags + tags[-diff:]
	return data, tags

def prepare_data(seqs, words):
	clipped_seqs = [seq[:CLIP] if len(seq) > CLIP else seq for seq in seqs]
	vectorized_seqs = [[words[tok] if tok in words else 1 for tok in seq] for seq in clipped_seqs]
	size = len(vectorized_seqs)
	data = []
	for i in range(0, int(size / BATCH_SIZE)):
		if i % 500 == 0:
			print("Prep Iteration = {}".format(i))
			sys.stdout.flush()
		start = i*BATCH_SIZE
		end = (i+1)*BATCH_SIZE
		data.append(pack_data(vectorized_seqs[start:end]))
	return data

def prepare_tags(seq, vocab):
	vectorized_tags = [vocab[tok] for tok in seq]
	size = len(vectorized_tags)
	data = []
	for i in range(0, int(size / BATCH_SIZE)):
		if i % 500 == 0:
			print("Prep Iteration = {}".format(i))
			sys.stdout.flush()
		start = i*BATCH_SIZE
		end = (i+1)*BATCH_SIZE
		if torch.cuda.is_available():
			data.append(Variable(torch.LongTensor(vectorized_tags[start:end]), requires_grad=False).cuda())
		else:
			data.append(Variable(torch.LongTensor(vectorized_tags[start:end]), requires_grad=False))
	return data

def pack_data(vectorized_seqs):
	# get the length of each seq in your batch
	seq_lengths = torch.LongTensor( list(map(len, vectorized_seqs)) )
	if torch.cuda.is_available():
		seq_tensor = Variable(torch.zeros((len(vectorized_seqs), CLIP)), requires_grad=False).long().cuda()
	else:
		seq_tensor = Variable(torch.zeros((len(vectorized_seqs), CLIP)), requires_grad=False).long()
	for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
		seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
	return seq_tensor

def accuracy(preds, rating, confusion):
	for pred, ref in zip(preds, rating):
		if ref == 0:
			if pred == ref:
				confusion[0][0] += 1
			elif pred == 1:
				confusion[0][1] += 1
			else:
				confusion[0][2] += 1
		elif ref == 1:
			if pred == ref:
				confusion[1][1] += 1
			elif pred == 0:
				confusion[1][0] += 1
			else:
				confusion[1][2] += 1
		else:
			if pred == ref:
				confusion[2][2] += 1
			elif pred == 1:
				confusion[2][1] += 1
			else:
				confusion[2][0] += 1
	return confusion

def output_preds(preds):
	pred_to_id = {0: 1, 1: 3, 2: 5}
	list_preds = []
	for tensor in preds:
		for pred in tensor:
			list_preds.append(pred_to_id[pred])
	list_preds = list_preds[:data_size]
	with open(outputfile, 'w') as f:
		for pred in list_preds:
			f.write(str(pred))
			f.write('\n')

''' Prediction Accuracy of model on validation dataset '''
def test():
	print("Validating")
	preds = []
	total, correct = 0, 0
	size = len(val_data)
	confusion = np.zeros((3,3))
	P = np.zeros(3)
	R = np.zeros(3)
	for i in range(0, size): 
		if i % 500 == 0:
			print("Iteration = {}".format(i))
		sentences_in = val_data[i]
		targets = val_tags[i]
		tag_scores = model(sentences_in.t())
		predicted = torch.max(tag_scores.data,1)[1]
		total += BATCH_SIZE
		correct += (predicted == targets.data).sum()
		confusion = accuracy(predicted, targets.data, confusion)
		preds.append(predicted)
	print("Prediction score = {}/{} = {}%".format(correct, total, float(100 * correct)/float(total)))
	for i in range(3):
		if confusion[i][i] == 0:
			P[i] = 0.0
			R[i] = 0.0
		else:
			R[i] = confusion[i][i] / (confusion[i][0] + confusion[i][1] + confusion[i][2])
			P[i] = confusion[i][i] / (confusion[0][i] + confusion[1][i] + confusion[2][i])
	FP = np.sum(P) / 3
	FR = np.sum(R) / 3
	FM = 2*FP*FR / (FP+FR)
	print("Precision = {}".format(FP))
	print("Recall = {}".format(FR))
	print("F-Measure = {}".format(FM))
	print(confusion)
	sys.stdout.flush()
	return preds

''' The Deep Neural Net Model '''
class Sentiment(nn.Module):
	def __init__(self, mode, embedding_dim, hidden_dim, vocab_size, tagset_size, vocab):
		super(Sentiment, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.model = mode
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)
		self.cell = self.init_model()
		self.hidden = self.init_hidden()
		self.vocab = vocab

	def init_model(self):
		if self.model == 'LSTM':
			bi_cell = torch.nn.LSTM(self.embedding_dim, self.hidden_dim, LAYERS, batch_first=False, bidirectional=True)
		elif self.model == 'GRU':
			bi_cell = torch.nn.GRU(self.embedding_dim, self.hidden_dim, LAYERS, batch_first=False, bidirectional=True)
		else:
			raise Exception('Specify one of following cell types: LSTM or GRU')
		return bi_cell

	def init_hidden(self):
		if torch.cuda.is_available():
			if self.model == 'LSTM':
				hidden = (Variable(torch.zeros(2*LAYERS, BATCH_SIZE, self.hidden_dim), requires_grad=False).cuda(),
							Variable(torch.zeros(2*LAYERS, BATCH_SIZE, self.hidden_dim), requires_grad=False).cuda())
			elif self.model == 'GRU':
				hidden = (Variable(torch.zeros(2*LAYERS, BATCH_SIZE, self.hidden_dim), requires_grad=False).cuda())
		else:
			if self.model == 'LSTM':
				hidden = (Variable(torch.zeros(2*LAYERS, BATCH_SIZE, self.hidden_dim), requires_grad=False),
							Variable(torch.zeros(2*LAYERS, BATCH_SIZE, self.hidden_dim), requires_grad=False))
			elif self.model == 'GRU':
				hidden = (Variable(torch.zeros(2*LAYERS, BATCH_SIZE, self.hidden_dim), requires_grad=False))
		return hidden

	def forward(self, sentences):
		embeds = self.word_embeddings(sentences)
		bi_output, bi_hidden = self.cell(embeds.view(-1, BATCH_SIZE,self.embedding_dim), self.hidden)
		tag_space = self.hidden2tag(bi_output[-1].view(BATCH_SIZE, -1))
		tag_scores = F.log_softmax(tag_space)
		return tag_scores

# Get data from files
testfile = str(sys.argv[1])
outputfile = str(sys.argv[2])
print("Reading")
read_start = time.time()
dev_text, val_tags, dev_summary = get_data(testfile)
print("Total Read Time = {}".format(time.time() - read_start))

val_data = [s*SUMMARY_WEIGHT + t for t, s in zip(dev_text, dev_summary)]
data_size = len(val_data)

# Retrieve Model
model = torch.load('neural_model_2014CS50462.txt')
word_to_ix = model.vocab
tag_to_ix = {1.0: 0, 3.0: 1, 5.0: 2}

sort_start = time.time()
(val_data, val_tags) = sort_data(val_data, val_tags)
print("Total Sort Time = {}".format(time.time() - sort_start))

val_data = prepare_data(val_data, word_to_ix)
val_tags = prepare_tags(val_tags, tag_to_ix)

# Test the model on the validation set
validation_start = time.time()
preds = test()
output_preds(preds)
print("Total Validation Time = {}".format(time.time() - validation_start))
