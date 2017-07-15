# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-03-28 17:13:02
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-05-01 22:49:35




import numpy as np
import re
f_ratio = 0

def load_data(input_file, FILTER=True):
	in_lines = open(input_file,'rU').readlines()
	whole_sentences = []
	single_sentence = {}
	base_words = []
	origin_words = []
	origin_collapse_words = []
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
			origin_collapse_words.append(line_dict['C']) 
			golden_labels.append(line_dict['G'])
			predict_labels.append(line_dict['P'])
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
				single_sentence['orig_collapse_word'] = origin_collapse_words
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
			origin_collapse_words = []
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
		single_sentence['orig_collapse_word'] = origin_collapse_words
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
	line_dict = {'word':None,'C':None, 'D':None,'G':None,'P':None,'V':None,'F':None,'T':None,'R':None,'label':None}
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
			line_dict['C'] = new_token
			line_dict['D'] = split_by(new_token, split_symbol)
		elif '[G]' in tokens[idx]:
			line_dict['G'] = new_token
		elif '[P]' in tokens[idx]:
			line_dict['P'] = new_token
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


def filter_duplicated_candidate(input_file, output_file):
	structures = load_data(input_file, True)
	example_num = len(structures)
	max_candidate = 11
	flatten_instance = []
	for idx in range(example_num):
		instance_num = max_candidate
		if len(structures[idx]) < max_candidate:
			instance_num = len(structures[idx])
		for idy in range(instance_num):
			flatten_instance.append(structures[idx][idy])
	print "Examples: ", example_num
	print "Instances: ", len(flatten_instance)
	out_file = open(output_file,'w')
	# print flatten_instance[14]
	for instance in flatten_instance:
		out_file.write(instance_to_string(instance))
		out_file.write('\n')
	out_file.close()


def instance_to_string(instance):
	out_string = ''
	sent_length = len(instance['input_word'])
	for idx in range(sent_length):
			out_string += instance['input_word'][idx]+' '
			out_string += '[D]'+ instance['orig_collapse_word'][idx] + ' '
			out_string += '[G]' + instance['gold_label'][idx] + ' '
			out_string += '[P]' + instance['pred_label'][idx] + ' '
			out_string += '[V]' + str(instance['accuracy']) + ' '
			out_string += '[F]' + str(instance['f1']) + ' '
			out_string += '[T]' + str(instance['tag']) + ' '
			out_string += '[R]' + str(instance['prob']) + ' '
			out_string += str(instance['label']) + '\n'
	return out_string


if __name__ == '__main__':

	train_file = "data/train.bio.rerank"
	dev_file = "data/dev.bio.rerank"
	test_file = "data/test.bio.rerank"
	input_file = train_file
	filter_duplicated_candidate(input_file, input_file.split('rerank')[0]+'filter')







	
