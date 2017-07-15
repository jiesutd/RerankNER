# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2016-11-24 11:57:41
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-07-15 17:13:30

import sys
import numpy as np
from utils.alphabet import Alphabet
from utils.data_processor import *
from model.Model import *
from utils.keras_utils import padding
from utils.metric import *
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



# from keras.utils.vis_utils import plot_model

word_alphabet = Alphabet('word')
char_alphabet = Alphabet('char')

nb_epoch = 100
use_char = True
mask_zero = True
BILSTM = True
DropProb = 0.2
case_sense = True
batch_size = 128
grad_discent = "adam"
lstm_average = False
label_type = 'BMES'
char_emb_dims = 50
nb_filter = 100
filter_length = 3
lstm_hidden_dims = 100
dense_hidden = 100
model_dict = { "A": "LSTM_model", "B": "LSTM_CNN_model", "C":"AttLSTM_model" }

def prepare_data(train_file, dev_file, test_file, embedding_file):
	np.random.seed(1337)
	create_alphabet([train_file, dev_file, test_file],word_alphabet, char_alphabet, True, True)

	pretrain_word_emb = match_embedding(embedding_file, word_alphabet)
	train_structure = load_rerank_data(train_file)
	train_word, train_char, train_label, train_mask = generate_tensor(train_structure, word_alphabet, char_alphabet, True, True, True)
	dev_structure = load_rerank_data(dev_file)
	dev_word, dev_char, dev_label, dev_mask = generate_tensor(dev_structure, word_alphabet, char_alphabet, True, True, False)
	test_structure = load_rerank_data(test_file)
	test_word, test_char, test_label, test_mask = generate_tensor(test_structure, word_alphabet, char_alphabet, True, True, False)

	print "Train instance:oracle:first =", len(train_word),":", oracle_best_f(train_structure, 11, label_type), ":", oracle_best_f(train_structure, 1, label_type)
	print "Dev instance:oracle:first =", len(dev_word),":", oracle_best_f(dev_structure, 10, label_type), ":", oracle_best_f(dev_structure, 1, label_type)
	print "Test instance:oracle:first =",len(test_word), ":",oracle_best_f(test_structure, 10, label_type), ":", oracle_best_f(test_structure, 1, label_type)

	return  train_word, train_char, train_label,\
			dev_word, dev_char, dev_label,\
			test_word, test_char, test_label,\
			pretrain_word_emb, \
			train_structure,dev_structure,test_structure


def main(TRAIN, FILE, MODEL_MODE, train_file, dev_file, test_file, embedding_file, selected_iter = -1):
	print ("Setting summary: ")
	print ("	MODEL_MODE: %s\n\
	USE CHAR:%s\n\
	AVGPOOL:%s\n\
	embedding_file: %s\n\
	nb_epoch: %s\n\
	mask_zero: %s\n\
	BILSTM: %s\n\
	DropProb: %s\n\
	case_sense: %s\n\
	grad_discent: %s\n\
	lstm_hidden_dims:%s\n\
	nb_filter:%s\n\
	char_emb_dims:%s" % (model_dict[MODEL_MODE],use_char,lstm_average,embedding_file,nb_epoch,mask_zero,BILSTM,DropProb, case_sense, grad_discent, lstm_hidden_dims, nb_filter, char_emb_dims))

	if "debug" in FILE.lower():
		print ("IN DEBUG MODEL......  IF TRAINING: %s" % TRAIN)
		DEBUG_MODEL = True
	else:
		print ("IN NORMAL MODEL...... IF TRAINING: %s" % TRAIN)
		DEBUG_MODEL = False


	if DEBUG_MODEL:
		train_file = "data/small_train.txt"
		dev_file = "data/small_dev.txt"
		test_file = "data/small_test.txt"

	print "	Train file:", train_file
	print "	Dev file:", dev_file
	print "	Test file:", test_file

	sys.stdout.flush()	
	X_train,X1_train,Y_train,X_dev,X1_dev,Y_dev,X_test,X1_test,Y_test,word_pretrain_embedding,structure_train,structure_dev,structure_test = prepare_data(train_file,dev_file,test_file, embedding_file)
	# set parameters:
	word_vocab_size = word_alphabet.size()
	char_vocab_size = char_alphabet.size()

	word_emb_dims = word_pretrain_embedding.shape[1]
	
	word_max_len = X1_train.shape[1]
	char_max_len = X1_train.shape[2]

	
	embedding_name = embedding_file.split('/')[-1]
	FILE_NAME = "./results/" +"MODEL."+MODEL_MODE+"_CHAR."+str(use_char)+"_DB."+str(DEBUG_MODEL)+"_Mask."+str(mask_zero)+"_BILSTM."+str(BILSTM) + "_Emb." + embedding_name + "_drop." + str(DropProb) +"_GD."+grad_discent+ "_Iter."

	def my_init(shape, dtype=None):
		return word_pretrain_embedding
	if MODEL_MODE == 'A':
		model = LSTM_model(use_char, word_vocab_size, word_max_len, char_vocab_size, char_max_len,  word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, my_init,mask_zero,BILSTM,DropProb,lstm_average,grad_discent)
	elif MODEL_MODE == 'B':
		model = LSTM_CNN_model(use_char, word_vocab_size, word_max_len, char_vocab_size, char_max_len,  word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, my_init,mask_zero,BILSTM,DropProb,lstm_average,grad_discent)
	elif MODEL_MODE == 'C':
		model = AttLSTM_model(use_char, word_vocab_size, word_max_len, char_vocab_size, char_max_len,  word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, my_init,mask_zero,BILSTM,DropProb,lstm_average,grad_discent)
	else:
		print "ERROR MODEL MODE:", MODEL_MODE
	# plot_model(model, to_file=MODEL_MODE+".BI."+str(BILSTM)+"."+model_dict[MODEL_MODE]+'.png',show_shapes=True)
	if use_char:
		train_in_dict = {"word_input":X_train,"char_input":X1_train}
		dev_in_dict = {"word_input":X_dev,"char_input":X1_dev}
		test_dict = {"word_input":X_test,"char_input":X1_test}
	else:
		train_in_dict = {"word_input":X_train}
		dev_in_dict = {"word_input":X_dev}
		test_dict = {"word_input":X_test}
	train_out_dict = {"output":Y_train}
	dev_out_dict = {"output":Y_dev}


	if TRAIN.lower()== "train":
		print ("Start to train model in normal step......")
		checkpointer = ModelCheckpoint(filepath= FILE_NAME+"{epoch:02d}.hdf5", verbose=1, save_best_only=False, mode='auto')
		model.fit(train_in_dict, train_out_dict, shuffle=True, epochs=nb_epoch, batch_size=batch_size, callbacks=[checkpointer], validation_data=(dev_in_dict, dev_out_dict) )
	
	## development data using all saved model
	best_f = -1
	best_epoch = -1
	best_accuracy = -1
	alpha_step = 0.005
	alpha_num = int(1/alpha_step)
	best_alpha = -1.0
	best_f = -1.0
	best_accuracy = -1.0

	print ("Start to load existing model %s......" % FILE_NAME)
	for idx in range(0, nb_epoch):
		if selected_iter >= 0:
			if idx != selected_iter:
				continue
			print "Select iteration:", selected_iter

		model_name = FILE_NAME + str(idx).zfill(2) + ".hdf5"
		model.load_weights(model_name)
		predict_dev = model.predict(dev_in_dict, batch_size, 0)
		epoch_accuracy = -1
		epoch_p = -1 
		epoch_r = -1
		epoch_f = -1
		epoch_alpha = -1
		for idy in range(alpha_num):
			alpha = idy * alpha_step
			golden_list, predict_choose_list = get_alpha_golden_predict_choose_results(structure_dev, predict_dev, alpha)
			accuracy = candidate_choose_accuracy(golden_list, predict_choose_list)
			p,r,f = get_rerank_ner_fmeasure(structure_dev,predict_choose_list,label_type)
			if f > epoch_f:
				epoch_f = f
				epoch_accuracy = accuracy
				epoch_p = p
				epoch_r = r
				epoch_alpha = alpha
				origin_dev_file = FILE_NAME+'origindev'
				save_predict_result(structure_dev, predict_choose_list, origin_dev_file)
		print ("epoch: %s; alpha:%s; best f: %s; choose accuracy:%s" % (idx, epoch_alpha, epoch_f, epoch_accuracy))
		if epoch_f > best_f:
			best_accuracy = epoch_accuracy
			best_alpha = epoch_alpha
			best_epoch = idx
			best_f = epoch_f
			best_p = epoch_p
			best_r = epoch_r
	if selected_iter >= 0:
		print ("Fix epoch/best alpha: %s/%s; P/R/F: %s/%s/%s;  choose accuracy: %s"% (best_epoch,best_alpha, best_p, best_r, best_f, best_accuracy))
	else:
		print ("Best epoch/alpha: %s/%s; P/R/F: %s/%s/%s;  choose accuracy: %s"% (best_epoch,best_alpha, best_p, best_r, best_f, best_accuracy))

	
	model_name = FILE_NAME + str(best_epoch).zfill(2) + ".hdf5"
	print ("Loading model: %s" % model_name)
	model.load_weights(model_name)

	predict_test = model.predict(test_dict, 32, 0)
	golden_list, predict_choose_list = get_alpha_golden_predict_choose_results(structure_test, predict_test, best_alpha)
	test_accuracy = candidate_choose_accuracy(golden_list, predict_choose_list)
	p,r,f = get_rerank_ner_fmeasure(structure_test,predict_choose_list,label_type)
	print ("Test data: P: %s , R: %s, F: %s, Accuracy: %s" % (p,r,f,test_accuracy))
	## save test result
	result_name = model_name.split('.hdf5')[0]
	save_predict_result(structure_test,predict_choose_list, result_name+'.test')
	## save dev result
	golden_list, predict_choose_list = get_alpha_golden_predict_choose_results(structure_dev, predict_dev, best_alpha)
	save_predict_result(structure_dev,predict_choose_list, result_name+'.dev')





if __name__ == '__main__':
	## set gpu usage in tensorflow backend
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.4
	set_session(tf.Session(config=config))


	embedding_file = "data/SENNA.emb"
	train_file = "data/train.rerank.filter"
	dev_file = "data/dev.rerank.filter"
	test_file = "data/test.rerank.filter"
	# train_file = "data/train.bio.filter"
	# dev_file = "data/dev.bio.filter"
	# test_file = "data/test.bio.filter"
	np.random.seed(1)
	# python main_margin.py train debug
	if len(sys.argv) > 4:
		main(sys.argv[1], sys.argv[2], sys.argv[3], train_file, dev_file, test_file, embedding_file, int(sys.argv[4]))
	elif len(sys.argv) == 4:
		main(sys.argv[1], sys.argv[2], sys.argv[3], train_file, dev_file, test_file, embedding_file)
	else:
		main(sys.argv[1],"NORMAL", "A", train_file, dev_file, test_file, embedding_file)










	
