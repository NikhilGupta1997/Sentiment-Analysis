import numpy as np
import json
from pprint import pprint
import string, re
from nltk.corpus import stopwords
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import sys
import os
import argparse
import subprocess

infile = sys.argv[1]
outfile = sys.argv[2]

summ_add = 5

def read_input(file):
	count = 0
	all_text = []
	all_ratings = []
	all_summaries = []
	for x in open(file):
		count += 1
		data = re.sub('"reviewText":|"overall":|"summary":', '@#^@', x).strip().split("@#^@")
		text = ','.join(data[1].split(',')[:-1]).strip().strip('\"')
		rating = float(data[2].split(',')[0].strip())
		summary = data[3].strip().strip('}').strip('\"')
		# all_text.append(parse_text(text)[0])
		# all_ratings.append(parse_rating(rating)[0])
		# all_summaries.append(parse_summary(summary)[0])
		all_text.append(parse_text(text))
		all_ratings.append(parse_rating(rating))
		all_summaries.append(parse_summary(summary))
	return all_text, all_ratings, all_summaries

def remove_punctuation(sentence):
	return sentence.strip(string.punctuation)

def remove_quot(sentence):
	return re.sub('quot;[^&]+&quot', '', sentence)

def remove_punc(sentence):
	s = "string. With. Punctuation"
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	return re.sub( '\s+', ' ', regex.sub(' ', sentence))

def remove_proper_nouns(sentence):
	return [word for i, word in enumerate(sentence.split()) if (not word.istitle() or i == 0)]

def to_lowercase(sentence):
	return " ".join([word.lower() for word in sentence.split()])

def remove_stopwords(sentence):
	return " ".join([wordnet_lemmatizer.lemmatize(word) for word in sentence if word not in stop_words])

def negation(sentence):
	for i, word in enumerate(sentence):
		if 'n\'t' in word or word == 'never' or word =='not' or word =='NOT' or word =='Not':
			if len(sentence) > i+1:
				sentence[i+1] = 'not_' + sentence[i+1]
			# if len(sentence) > i+2:
			# 	sentence[i+2] = 'not_' + sentence[i+2]
	return sentence

def parse_text(text):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	pos = ['CC', 'DT', 'JJ', 'JJR', 'JJS', 'MD', 'PDT', 'RB', 'RBR', 'RBS', 'VBN', 'VBZ','WRB']
	# for i, story in enumerate(text):
	sentences = tokenizer.tokenize(text)
	for j, sentence in enumerate(sentences):
		# sentence = remove_punctuation(sentence)
		sentence = remove_quot(sentence)
		# tagged_sentence = nltk.tag.pos_tag(sentence.split())
		# edited_sentence = [word.strip() for word,tag in tagged_sentence if tag in pos]
		# sentence = ' '.join(edited_sentence)
		# sentence = to_lowercase(sentence)
		sentence = remove_proper_nouns(sentence)
		sentence = negation(sentence)
		# sentence = remove_stopwords(sentence)
		sentences[j] = ' '.join(sentence)
	text = "".join(sentences)
	return text

def parse_rating(score):
	if score < 3.0:
		return 1.0
	elif score > 3.0:
		return 5.0
	return 3.0

def parse_summary(sentence):
	pos = ['CC', 'DT', 'JJ', 'JJR', 'JJS', 'MD', 'PDT', 'RB', 'RBR', 'RBS', 'VBN', 'VBZ','WRB']
	# sentence = remove_punctuation(sentence)
	sentence = remove_quot(sentence)
	# sentence = remove_punc(sentence)
	# tagged_sentence = nltk.tag.pos_tag(sentence.split())
	# edited_sentence = [word for word,tag in tagged_sentence if tag in pos]
	# sentence = ' '.join(edited_sentence)
	# sentence = to_lowercase(sentence)
	sentence = remove_proper_nouns(sentence)
	sentence = ' '.join(negation(sentence))
	# sentence = remove_stopwords(sentence)
	return sentence

def accuracy(preds, rating):
	confusion = np.zeros((3,3))
	P = np.zeros(3)
	R = np.zeros(3)
	for pred, ref in zip(preds, rating):
		if ref == 1.0:
			if pred == ref:
				confusion[0][0] += 1
			elif pred == 3.0:
				confusion[0][1] += 1
			else:
				confusion[0][2] += 1
		elif ref == 3.0:
			if pred == ref:
				confusion[1][1] += 1
			elif pred == 1.0:
				confusion[1][0] += 1
			else:
				confusion[1][2] += 1
		else:
			if pred == ref:
				confusion[2][2] += 1
			elif pred == 3.0:
				confusion[2][1] += 1
			else:
				confusion[2][0] += 1
	for i in range(3):
		R[i] = confusion[i][i] / (confusion[i][0] + confusion[i][1] + confusion[i][2])
		if confusion[i][i] == 0:
			P[i] = 0.0
		else:
			P[i] = confusion[i][i] / (confusion[0][i] + confusion[1][i] + confusion[2][i])
	FP = np.sum(P) / 3.0
	FR = np.sum(R) / 3.0
	FM = 2.0*FP*FR / (FP+FR)
	return FP, FR, FM, confusion

dev_text, dev_rating, dev_summary = read_input(infile)
val_text = [t + s*summ_add for t, s in zip(dev_text, dev_summary)]

with open('2014CS50462_model.txt', 'rb') as modelfile:
	clf = pickle.load(modelfile)

preds = clf.predict(val_text)

P, R, F, C = accuracy(preds, dev_rating)
print(C)
print(P, R)
print('F-score = ', F)

with open(outfile, 'w') as file:
	for pred in preds:
		file.write("%d\n" % pred)

