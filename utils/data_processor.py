# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-03-28 17:13:02
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-05-01 21:17:51




import numpy as np
import re
from alphabet import Alphabet
word_end = "##WE##"
MAX_WORD_LENGTH = 130
MAX_CHAR_LENGTH = 45
f_ratio = 0.5



def create_alphabet(files, word_alphabet, char_alphabet, case_sense=True, norm_digit = True):
	char_alphabet.add(word_end)
	print ("Create alphabets, case_sense:%s, norm_digit:%s" %(case_sense,norm_digit))
	if len(files) <= 0:
		print "Create alphabet error, should fill input file list!"
	assert(len(files) > 0 )
	for file in files:
		in_lines = open(file,'r').readlines()
		for line in in_lines:
			if len(line) > 2:
				word = line.split(' ')[0]
				if norm_digit:
					word = normalized_digit(word)
				for idx in range(len(word)):
					char_alphabet.add(word[idx])
				if not case_sense:
					word = word.lower()
				word_alphabet.add(word)
	word_alphabet.close()
	char_alphabet.close()
	print "Word Alphabet size:", word_alphabet.size()
	print "Char Alphabet size:", char_alphabet.size()


def load_embedding(embedding_file):
	embedding_lines = open(embedding_file,'r').readlines()
	embedding_words = []
	embeddings = []
	for each_embedding in embedding_lines:
		seperate_embedding = each_embedding.strip(' \n').split(' ')	
		embedding_words.append(seperate_embedding[0])
		embeddings.append(np.asarray(map(float,seperate_embedding[1:]) ) )
	return embedding_words, embeddings


def match_embedding(embedding_file, the_alphabet, default_dim = 30):
	if embedding_file == None:
		print "Match embedding: no embedding file found, random initialized."
		embed_words, embed_values = [{}, {}]
		embed_num = 0
		embed_dim = default_dim
	else:
		embed_words, embed_values = load_embedding(embedding_file)
		embed_num = len(embed_words)
		embed_dim = len(embed_values[0])

	alphabet_size = the_alphabet.size()
	scale = np.sqrt(3.0 / embed_dim)
	embed_table = np.empty([the_alphabet.size(), embed_dim])
	out_of_vocab = 0
	direct_match = 0
	lower_match = 0
	for word, index in the_alphabet.iteritems():
		if word in embed_words:
			embed_table[index, :] = embed_values[embed_words.index(word)]
			direct_match += 1
		elif word.lower() in embed_words:
			embed_table[index, :] = embed_values[embed_words.index(word.lower())]
			lower_match += 1
		else:
			embed_table[index, :] = np.random.uniform(-scale, scale, [1, embed_dim])
			out_of_vocab += 1
	print("Embedding file number: %s; dim:%s"%(embed_num, embed_dim))
	print ("Alphabet size: %s; direct_match:%s; lower_match:%s; OOV_num:%s; OOV:%s"%(alphabet_size, direct_match, lower_match, out_of_vocab, (out_of_vocab+0.)/alphabet_size))
	return embed_table


def normalized_digit(word):
	return re.sub('\d', '0', word)


def load_rerank_data(input_file, FILTER=True):
	in_lines = open(input_file,'rU').readlines()
	whole_sentences = []
	single_sentence = {}
	base_words = []
	origin_words = []
	golden_labels = []
	predict_labels = []
	value = -1.0
	f1 = -1.0
	tag = "-1"
	prob = 0.0
	label = False
	target_score = -1.0
	F_ratio = f_ratio
	for each_line in in_lines:
		if len(each_line) > 2:
			line_dict = line2dict(each_line)
			word = line_dict['word']
			base_words.append(word)
			origin_words += line_dict['D']
			golden_labels += line_dict['G']
			predict_labels += line_dict['P']
			value = line_dict['V']
			if line_dict['F'] != None:
				f1 = line_dict['F']
				target_score = value * (1-F_ratio) + f1*F_ratio
			else:
				target_score = value
				F_ratio = 0
			tag = line_dict['T']
			prob = line_dict['R']
			label = line_dict['label']
			
		else:
			if len(base_words) != 0:
				single_sentence['input_word'] = base_words
				single_sentence['orig_word'] = origin_words
				single_sentence['gold_label'] = golden_labels
				single_sentence['pred_label'] = predict_labels
				single_sentence['accuracy'] = value
				single_sentence['f1'] = f1
				single_sentence['tag'] = tag
				single_sentence['prob'] = prob
				single_sentence['label'] = label
				single_sentence['target'] = target_score
				whole_sentences.append(single_sentence)
			base_chars = []
			single_sentence = {}
			base_words = []
			origin_words = []
			golden_labels = []
			predict_labels = []
			value = -1.0
			f1 = -1.0
			tag = "-1"
			prob = 0.0
			label = False
			target_score = -1.0
	if base_words:
		single_sentence['input_word'] = base_words
		single_sentence['orig_word'] = origin_words
		single_sentence['gold_label'] = golden_labels
		single_sentence['pred_label'] = predict_labels
		single_sentence['accuracy'] = value
		single_sentence['f1'] = f1
		single_sentence['tag'] = tag
		single_sentence['prob'] = prob
		single_sentence['label'] = label
		single_sentence['target'] = target_score
		whole_sentences.append(single_sentence)

	structure_sentences = []
	each_candidates = []
	current_list = whole_sentences[0]['orig_word']
	current_input_lists = []
	candidate_num = 10
	different_candidate_list = []
	for sentence in whole_sentences:
		if sentence['orig_word'] == current_list:
			if FILTER:
				if sentence['input_word'] in different_candidate_list:
					continue
			different_candidate_list.append(sentence['input_word'])
			each_candidates.append(sentence)
		else:
			structure_sentences.append(each_candidates)
			each_candidates = []
			each_candidates.append(sentence)
			current_list = sentence['orig_word']
			different_candidate_list = []
			different_candidate_list.append(sentence['input_word'])
	if each_candidates:
		structure_sentences.append(each_candidates)
	print "F_ratio:", F_ratio
	return structure_sentences


def line2dict(input_line):
	tokens = input_line.strip('\n').split(' ')
	token_num = len(tokens)
	line_dict = {'word':None, 'D':None,'G':None,'P':None,'V':None,'F':None,'T':None,'R':None,'label':None}
	line_dict['word'] = tokens[0]
	if tokens[-1] == "True":
		line_dict['label'] = True
	else:
		line_dict['label'] = False
	split_symbol = "*#*"
	for idx in range(1, token_num-1):
		## remove first 3 character
		new_token = tokens[idx][3:]
		if '[D]' in tokens[idx]:
			line_dict['D'] = split_by(new_token, split_symbol)
		elif '[G]' in tokens[idx]:
			line_dict['G'] = split_by(new_token, split_symbol)
		elif '[P]' in tokens[idx]:
			line_dict['P'] = split_by(new_token, split_symbol)
		elif '[V]' in tokens[idx]:
			line_dict['V'] = round(float(new_token),4)
		elif '[F]' in tokens[idx]:
			line_dict['F'] = round(float(new_token),4)
		elif '[T]' in tokens[idx]:
			line_dict['T'] = round(float(new_token),4)
		elif '[R]' in tokens[idx]:
			line_dict['R'] = round(float(new_token),4)
		else:
			print "More information provides:", tokens[idx]
	return line_dict


def split_by(input_line, symbol="*#*"):
	if symbol in input_line:
		return input_line.split(symbol)
	else:
		return [input_line]


def generate_tensor(structure_sentences, word_alphabet, char_alphabet, case_sense=True, norm_digit=True, training=False):
	words_array = []
	chars_array = []
	label_array = []
	
	for each_candidates in structure_sentences:
		max_score = 0.0
		min_score = 1.0
		for each_sentence in each_candidates:
			if each_sentence['target'] > max_score:
				max_score = float(each_sentence['target'])
			if each_sentence['target'] < min_score:
				min_score = each_sentence['target']
		mid_score = (min_score+max_score)/2
		score_range = max_score-min_score
		if score_range == 0:
			print "max score:",max_score, "min score:", min_score, " num:", len(each_candidates), 
			if training:
				print '; during training, ignore...'
				# print each_candidates
				continue
			factor = 1.0
		else:
			factor = 1.0/score_range
		for each_sentence in each_candidates:
			word_Ids = []
			char_Ids = []
			for word in each_sentence['input_word']:
				if norm_digit:
					word = normalized_digit(word)
				one_char_Ids = []
				for char in word[:MAX_CHAR_LENGTH]:
					one_char_Ids.append(char_alphabet.get_index(char))
				char_Ids.append(one_char_Ids)
				if not case_sense:
					word = word.lower()
				word_Ids.append(word_alphabet.get_index(word))
			label_array.append((each_sentence['target']-mid_score)*factor+0.5)
			words_array.append(word_Ids)
			# print "char length:", len(char_Ids)
			chars_array.append(char_Ids)

	## now we have lists: words_array, chars_array, label_array
	instance_num = len(words_array)
	print "Instance size:", instance_num
	## generate label numpy tensor
	label_tensor = np.asarray(label_array)

	## generate word and char numpy tensor
	word_end_id = char_alphabet.get_index(word_end)
	char_tensor = np.empty([instance_num, MAX_WORD_LENGTH, MAX_CHAR_LENGTH], dtype=np.int32)
	word_tensor = np.empty([instance_num, MAX_WORD_LENGTH], dtype=np.int32)
	word_mask = np.zeros([instance_num, MAX_WORD_LENGTH])
	for idx in range(instance_num):
		chars_ids = chars_array[idx]
		word_ids = words_array[idx]
		word_length = len(word_ids)
		for idy in range(MAX_WORD_LENGTH):
			if idy >= word_length:
				word_tensor[idx, idy] = 0
				char_tensor[idx, idy, :] = word_end_id
				word_mask[idx, idy] = 1
			else:
				word_tensor[idx,idy] = word_ids[idy]
				each_word_size  = len(chars_ids[idy])
				for idz in range(MAX_CHAR_LENGTH):
					if idz >= each_word_size:
						char_tensor[idx, idy, idz] = word_end_id
					else:
						char_tensor[idx, idy, idz] = chars_array[idx][idy][idz]

	## finish generate word, char, label tensor and word mask
	return word_tensor, char_tensor, label_tensor, word_mask

def generate_tensor_char_sep(structure_sentences, word_alphabet, char_alphabet,MAX_WORD_LENGTH=1,MAX_CHAR_LENGTH=1,case_sense=True, norm_digit=True, training=False):
	words_array = []
	chars_array = []
	label_array = []
	max_word_num = 0
	max_char_num = 0
	for each_candidates in structure_sentences:
		max_score = 0.0
		min_score = 1.0
		for each_sentence in each_candidates:
			if each_sentence['target'] > max_score:
				max_score = float(each_sentence['target'])
			if each_sentence['target'] < min_score:
				min_score = each_sentence['target']
		mid_score = (min_score+max_score)/2
		score_range = max_score-min_score
		if score_range == 0:
			print "max score:",max_score, "min score:", min_score, " num:", len(each_candidates), 
			if training:
				print '; during training, ignore...'
				continue
			factor = 1.0
		else:
			factor = 1.0/score_range
 		for each_sentence in each_candidates:
			word_Ids = []
			char_Ids = []
			for word in each_sentence['input_word']:
				if norm_digit:
					word = normalized_digit(word)
				for char in word:
					char_Ids.append(char_alphabet.get_index(char))
				if not case_sense:
					word = word.lower()
				word_Ids.append(word_alphabet.get_index(word))
			if len(word_Ids) > max_word_num:
				max_word_num = len(word_Ids)
			if len(char_Ids) > max_char_num:
				max_char_num = len(char_Ids)
			label_array.append((each_sentence['target']-mid_score)*factor+0.5)
			words_array.append(word_Ids)
			# print "char length:", len(char_Ids)
			chars_array.append(char_Ids)
	if training:
		MAX_WORD_LENGTH = max_word_num
		MAX_CHAR_LENGTH = max_char_num
	## now we have lists: words_array, chars_array, label_array
	instance_num = len(words_array)
	print "Instance size:", instance_num
	## generate label numpy tensor
	label_tensor = np.asarray(label_array)

	## generate word and char numpy tensor
	word_end_id = char_alphabet.get_index(word_end)
	char_tensor = np.empty([instance_num, MAX_CHAR_LENGTH], dtype=np.int32)
	word_tensor = np.empty([instance_num, MAX_WORD_LENGTH], dtype=np.int32)
	word_mask = np.zeros([instance_num, MAX_WORD_LENGTH])
	for idx in range(instance_num):
		chars_Ids = chars_array[idx]
		word_Ids = words_array[idx]
		word_length = len(word_Ids)
		char_length = len(char_Ids)
		for idy in range(MAX_WORD_LENGTH):
			if idy >= word_length:
				word_tensor[idx, idy] = 0
				word_mask[idx, idy] = 1
			else:
				word_tensor[idx,idy] = word_Ids[idy]
		for idy in range(MAX_CHAR_LENGTH):
			if idy >= char_length:
				char_tensor[idx, idy] = 0
			else:
				char_tensor[idx, idy] = char_Ids[idy]
				
	return word_tensor, char_tensor, label_tensor, word_mask, max_word_num, max_char_num


def filter_duplicated_candidate(input_file, output_file):
	structures = load_rerank_data(input_file, True)
	example_num = len(structures)
	flatten_instance = []
	for idx in range(example_num):
		for idy in range(len(structures[idx])):
			flatten_instance.append(structures[idx][idy])
	print "Examples: ", example_num
	print "Instances: ", len(flatten_instance)
	out_file = open(output_file,'w')
	for instance in flatten_instance:
		out_file.write(instance_to_string(instance))
		out_file.write('\n')
	out_file.close()


def instance_to_string(instance):
	out_string = ''
	sent_length = len(instance['input_word'])
	for idx in range(sent_length):
			out_string += instance['input_word'][idx]+' '
			out_string += '[D]'+ instance['orig_word'][idx] + ' '
			out_string += '[G]' + instance['gold_label'][idx] + ' '
			out_string += '[P]' + instance['pred_label'][idx] + ' '
			out_string += '[V]' + str(instance['accuracy']) + ' '
			out_string += '[F]' + str(instance['f1']) + ' '
			out_string += '[T]' + str(instance['tag']) + ' '
			out_string += '[R]' + str(instance['prob']) + ' '
			out_string += str(instance['label']) + '\n'
	return out_string


if __name__ == '__main__':

	train_file = "../data/train.bio.rerank"
	dev_file = "../data/dev.bio.rerank"
	test_file = "../data/test.bio.rerank"
	embed_file = None
	filter_duplicated_candidate(test_file, "../data/test.bio.filter")
	# embed_file = '../data/SENNA.emb'
	# word_alphabet = Alphabet('word')
	# char_alphabet = Alphabet('char')
	# create_alphabet([train_file,dev_file, test_file],word_alphabet, char_alphabet, True, True)
	# pretrain_word_emb = match_embedding(embed_file, word_alphabet)
	# train_structure = load_rerank_data(train_file)
	# train_word, train_char, train_label, train_mask = generate_tensor(train_structure, word_alphabet, char_alphabet, True, True)






	
