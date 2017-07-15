# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2016-11-23 21:21:17
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-04-29 22:18:18

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Model
# from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Dense, Reshape, Permute
from keras.layers import LSTM,GRU, Input
from keras.layers import GaussianDropout,Merge
from keras import initializers
from keras.layers import TimeDistributed
from keras.layers import Concatenate,RepeatVector,Multiply
# from keras.layers import Bidirectional
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,Flatten, AveragePooling1D, Bidirectional
from keras.regularizers import l2
from model.zeromasking import ZeroMaskedEntries

from keras import backend as K
# from softattention import Attention


char_filter = 50
char_filter_length = 3

def LSTM_model(use_char, word_vocab_size, word_max_len, char_vocab_size, char_max_len, word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, word_pretrain_embedding,mask_zero,BILSTM, drop_prob,lstm_average, update="adam"):
	## word emb
	word_input = Input(shape=(word_max_len, ), dtype="int32", name='word_input')
	word_emb = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=mask_zero)(word_input)

	if use_char:
		char_input = Input(shape=(word_max_len, char_max_len, ), dtype="int32", name='char_input')
		char_reshape0 = Reshape((word_max_len*char_max_len, ))(char_input)
		char_emb = Embedding(input_dim=char_vocab_size, output_dim=char_emb_dims, input_length=word_max_len*char_max_len, mask_zero=False)(char_reshape0)
		# TODO: add dropout
		## char CNN
		char_reshape = Reshape((word_max_len, char_max_len, char_emb_dims))(char_emb)
		char_conv = TimeDistributed(Conv1D(char_filter,char_filter_length, padding='valid'))(char_reshape)
		char_pooling = TimeDistributed(MaxPooling1D(char_max_len - char_filter_length + 1))(char_conv)
		char_rep = Reshape((word_max_len, -1))(char_pooling)
		word_rep = Concatenate()([char_rep, word_emb])
		word_drop = GaussianDropout(drop_prob)(word_rep)
	else:
		word_drop = GaussianDropout(drop_prob)(word_emb)
		
	## add lstm
	if BILSTM:
		word_lstm = Bidirectional(LSTM(lstm_hidden_dims))(word_drop)
	else:
		word_lstm = LSTM(lstm_hidden_dims)(word_drop)
	## TODO: dropout
	## final input
	final_hidden = Dense(dense_hidden, activation='relu')(word_lstm)
	main_output = Dense(1, activation='sigmoid',name='output')(final_hidden)
	if use_char:
		model = Model(inputs=[word_input, char_input], outputs=[main_output])
	else:
		model = Model(inputs=[word_input], outputs=[main_output])
	model.compile(loss='mse',optimizer=update,metrics=['accuracy'])
	return model



def LSTM_CNN_model(use_char, word_vocab_size, word_max_len, char_vocab_size, char_max_len, word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, word_pretrain_embedding,mask_zero,BILSTM, drop_prob,lstm_average, update="adam"):
	# word emb
	word_input = Input(shape=(word_max_len, ), dtype="int32", name='word_input')
	word_cnn_emb = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=False)(word_input)
	word_lstm_emb = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=mask_zero)(word_input)
	if use_char:
		char_input = Input(shape=(word_max_len, char_max_len, ), dtype="int32", name='char_input')
		char_reshape0 = Reshape((word_max_len*char_max_len, ))(char_input)
		char_emb = Embedding(input_dim=char_vocab_size, output_dim=char_emb_dims, input_length=word_max_len*char_max_len, mask_zero=False)(char_reshape0)
		# TODO: add dropout
		## char CNN
		char_reshape = Reshape((word_max_len, char_max_len, char_emb_dims))(char_emb)
		char_conv = TimeDistributed(Conv1D(char_filter, char_filter_length, padding='valid'))(char_reshape)
		char_pooling = TimeDistributed(MaxPooling1D(char_max_len - char_filter_length + 1))(char_conv)
		char_rep = Reshape((word_max_len, -1))(char_pooling)
		word_cnn_rep = Concatenate()([char_rep, word_cnn_emb])
		word_cnn_drop = GaussianDropout(drop_prob)(word_cnn_rep)
		word_lstm_rep = Concatenate()([char_rep, word_lstm_emb])
		word_lstm_drop = GaussianDropout(drop_prob)(word_lstm_rep)
	else:
		word_cnn_drop = GaussianDropout(drop_prob)(word_cnn_emb)
		word_lstm_drop = GaussianDropout(drop_prob)(word_lstm_emb)

	## add lstm
	if BILSTM:
		word_lstm = Bidirectional(LSTM(lstm_hidden_dims))(word_lstm_drop)
	else:
		word_lstm = LSTM(lstm_hidden_dims)(word_lstm_drop)	
	## add cnn
	word_cnn = Conv1D(filters=nb_filter,kernel_size=filter_length,activation='relu')(word_cnn_drop)
	word_cnn_pool = MaxPooling1D(pool_size=word_max_len-filter_length+1)(word_cnn)
	word_cnn_flat = Flatten()(word_cnn_pool)

	## concat all and dropout
	concat_lstm = Concatenate()([word_lstm, word_cnn_flat])

	## final input
	final_hidden = Dense(dense_hidden, activation='relu')(concat_lstm)
	main_output = Dense(1, activation='sigmoid',name='output')(final_hidden)
	if use_char:
		model = Model(inputs=[word_input, char_input], outputs=[main_output])
	else:
		model = Model(inputs=[word_input], outputs=[main_output])
	model.compile(loss='mse',optimizer=update,metrics=['accuracy'])
	return model


def AttLSTM_model(use_char, word_vocab_size, word_max_len, char_vocab_size, char_max_len, word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, word_pretrain_embedding,mask_zero,BILSTM, drop_prob,lstm_average, update="adam"):
	word_input = Input(shape=(word_max_len, ), dtype="int32", name='word_input')
	word_emb = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=mask_zero)(word_input)
	zero_word = ZeroMaskedEntries()(word_emb)
	if use_char:
		char_input = Input(shape=(word_max_len, char_max_len, ), dtype="int32", name='char_input')
		char_reshape0 = Reshape((word_max_len*char_max_len, ))(char_input)
		char_emb = Embedding(input_dim=char_vocab_size, output_dim=char_emb_dims, input_length=word_max_len*char_max_len, mask_zero=False)(char_reshape0)
		# TODO: add dropout
		## char CNN
		char_reshape = Reshape((word_max_len, char_max_len, char_emb_dims))(char_emb)
		char_conv = TimeDistributed(Conv1D(char_filter,char_filter_length, padding='valid'))(char_reshape)
		char_pooling = TimeDistributed(MaxPooling1D(char_max_len - char_filter_length + 1))(char_conv)
		char_rep = Reshape((word_max_len, -1))(char_pooling)
		## word emb
		word_rep = Concatenate()([char_rep, zero_word])
		word_drop = GaussianDropout(drop_prob)(word_rep)
	else:
		word_drop = GaussianDropout(drop_prob)(zero_word)
	## add lstm
	if BILSTM:
		word_lstm = Bidirectional(LSTM(lstm_hidden_dims,return_sequences=True),merge_mode='concat')(word_drop) 
	else:
		word_lstm = LSTM(lstm_hidden_dims,return_sequences=True)(word_drop)
	att = TimeDistributed(Dense(1))(word_lstm)
	att = Flatten()(att)
	att = Activation(activation="softmax")(att)
	att = RepeatVector(lstm_hidden_dims)(att)
	att = Permute((2,1))(att)
	mer = Multiply()([att, word_lstm])
	hid = AveragePooling1D(pool_size=word_max_len)(mer)
	attention = Flatten()(hid)
	## TODO: dropout
	## final input
	final_hidden = Dense(dense_hidden, activation='relu')(attention)
	main_output = Dense(1, activation='sigmoid',name='output')(final_hidden)
	if use_char:
		model = Model(inputs=[word_input, char_input], outputs=[main_output])
	else:
		model = Model(inputs=[word_input], outputs=[main_output])
	model.compile(loss='mse',optimizer=update,metrics=['accuracy'])
	return model
