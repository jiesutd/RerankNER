# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2016-11-23 21:21:17
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-04-24 00:57:35

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Dense
from keras.layers import LSTM,GRU, Input
from keras.layers import GaussianDropout,concatenate


# from keras.layers import Bidirectional
from keras.layers import Merge, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,Flatten, AveragePooling1D
from keras.regularizers import l2
from keras import backend as K


def Graph_Full_model(word_vocab_size, word_max_len, char_vocab_size, char_max_len, word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, word_pretrain_embedding,mask_zero,BILSTM, drop_prob,lstm_average, update="adam"):
	word_input = Input(shape=(word_max_len, ), dtype="int32", name='word_input')
	char_input = Input(shape=(char_max_len, ), dtype="int32", name='char_input')

	word_emb = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=mask_zero)(word_input)
	char_emb = Embedding(input_dim=char_vocab_size, output_dim=char_emb_dims, input_length=char_max_len, mask_zero=mask_zero)(char_input)

	word_emb_cnn = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=False)(word_input)
	char_emb_cnn = Embedding(input_dim=char_vocab_size, output_dim=char_emb_dims, input_length=char_max_len, mask_zero=False)(char_input)

	word_drop = GaussianDropout(drop_prob)(word_emb)
	char_drop = GaussianDropout(drop_prob)(char_emb)

	word_cnn_drop = GaussianDropout(drop_prob)(word_emb_cnn)
	char_cnn_drop = GaussianDropout(drop_prob)(char_emb_cnn)
	## add lstm
	word_lstm = LSTM(lstm_hidden_dims)(word_drop)
	char_lstm = LSTM(lstm_hidden_dims)(char_drop)
	
	## add cnn
	word_cnn = Conv1D(filters=nb_filter,kernel_size=filter_length,activation='relu')(word_cnn_drop)
	char_cnn = Conv1D(filters=nb_filter,kernel_size=filter_length,activation='relu')(char_cnn_drop)
	word_cnn_pool = MaxPooling1D(pool_size=5)(word_cnn)
	char_cnn_pool = MaxPooling1D(pool_size=5)(char_cnn)
	word_cnn_flat = Flatten()(word_cnn_pool)
	char_cnn_flat = Flatten()(char_cnn_pool)

	## concat all and dropout
	concat_lstm = concatenate([word_lstm, char_lstm, word_cnn_flat, char_cnn_flat])
	concat_noise = GaussianDropout(drop_prob)(concat_lstm)

	## final input
	final_hidden = Dense(dense_hidden, activation='relu')(concat_noise)
	main_output = Dense(1, activation='sigmoid',name='output')(final_hidden)
	model = Model(inputs=[word_input, char_input], outputs=[main_output])
	model.compile(loss='mse',optimizer=update,metrics=['accuracy'])
	return model


def Graph_LSTM_model(word_vocab_size, word_max_len, char_vocab_size, char_max_len, word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, word_pretrain_embedding,mask_zero,BILSTM, drop_prob,lstm_average, update="adam"):
	word_input = Input(shape=(word_max_len, ), dtype="int32", name='word_input')
	char_input = Input(shape=(char_max_len, ), dtype="int32", name='char_input')
	word_emb = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=mask_zero)(word_input)
	char_emb = Embedding(input_dim=char_vocab_size, output_dim=char_emb_dims, input_length=char_max_len, mask_zero=mask_zero)(char_input)
	word_drop = GaussianDropout(drop_prob)(word_emb)
	char_drop = GaussianDropout(drop_prob)(char_emb)
	word_lstm = LSTM(lstm_hidden_dims)(word_drop)
	char_lstm = LSTM(lstm_hidden_dims)(char_drop)
	concat_lstm = concatenate([word_lstm, char_lstm])
	concat_noise = GaussianDropout(drop_prob)(concat_lstm)
	final_hidden = Dense(dense_hidden, activation='relu')(concat_noise)
	main_output = Dense(1, activation='sigmoid',name='output')(final_hidden)
	model = Model(inputs=[word_input, char_input], outputs=[main_output])
	model.compile(loss='mse',optimizer=update,metrics=['accuracy'])
	return model


def Graph_Word_LSTM_Word_CNN_model(word_vocab_size, word_max_len, char_vocab_size, char_max_len, word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, word_pretrain_embedding,mask_zero,BILSTM, drop_prob,lstm_average, update="adam"):
	word_input = Input(shape=(word_max_len, ), dtype="int32", name='word_input')
	word_emb = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=mask_zero)(word_input)

	word_emb_cnn = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=False)(word_input)

	word_drop = GaussianDropout(drop_prob)(word_emb)
	word_cnn_drop = GaussianDropout(drop_prob)(word_emb_cnn)
	## add lstm
	word_lstm = LSTM(lstm_hidden_dims)(word_drop)
	
	## add cnn
	word_cnn = Conv1D(filters=nb_filter,kernel_size=filter_length,activation='relu')(word_cnn_drop)
	word_cnn_pool = MaxPooling1D(pool_size=5)(word_cnn)
	word_cnn_flat = Flatten()(word_cnn_pool)

	## concat all and dropout
	concat_lstm = concatenate([word_lstm,  word_cnn_flat])
	concat_noise = GaussianDropout(drop_prob)(concat_lstm)

	## final input
	final_hidden = Dense(dense_hidden, activation='relu')(concat_noise)
	main_output = Dense(1, activation='sigmoid',name='output')(final_hidden)
	model = Model(inputs=[word_input], outputs=[main_output])
	model.compile(loss='mse',optimizer=update,metrics=['accuracy'])
	return model

def Graph_Word_LSTM_Char_CNN_model(word_vocab_size, word_max_len, char_vocab_size, char_max_len, word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, word_pretrain_embedding,mask_zero,BILSTM, drop_prob,lstm_average, update="adam"):
	word_input = Input(shape=(word_max_len, ), dtype="int32", name='word_input')
	char_input = Input(shape=(char_max_len, ), dtype="int32", name='char_input')

	word_emb = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=mask_zero)(word_input)
	char_emb = Embedding(input_dim=char_vocab_size, output_dim=char_emb_dims, input_length=char_max_len, mask_zero=False)(char_input)

	word_drop = GaussianDropout(drop_prob)(word_emb)
	char_drop = GaussianDropout(drop_prob)(char_emb)
	## add lstm
	word_lstm = LSTM(lstm_hidden_dims)(word_drop)
	
	## add cnn
	char_cnn = Conv1D(filters=nb_filter,kernel_size=filter_length,activation='relu')(char_drop)
	char_cnn_pool =  MaxPooling1D(pool_size=5)(char_cnn)
	char_cnn_flat = Flatten()(char_cnn_pool)

	## concat all and dropout
	concat_lstm = concatenate([word_lstm, char_cnn_flat])
	concat_noise = GaussianDropout(drop_prob)(concat_lstm)

	## final input
	final_hidden = Dense(dense_hidden, activation='relu')(concat_noise)
	main_output = Dense(1, activation='sigmoid',name='output')(final_hidden)
	model = Model(inputs=[word_input, char_input], outputs=[main_output])
	model.compile(loss='mse',optimizer=update,metrics=['accuracy'])
	return model


def Graph_Word_LSTM_model(word_vocab_size, word_max_len, char_vocab_size, char_max_len, word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, word_pretrain_embedding,mask_zero,BILSTM, drop_prob,lstm_average, update="adam"):
	word_input = Input(shape=(word_max_len, ), dtype="int32", name='word_input')
	char_input = Input(shape=(char_max_len, ), dtype="int32", name='char_input')

	word_emb = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=mask_zero)(word_input)

	word_drop = GaussianDropout(drop_prob)(word_emb)
	## add lstm
	word_lstm = LSTM(lstm_hidden_dims)(word_drop)
	concat_noise = GaussianDropout(drop_prob)(word_lstm)

	## final input
	final_hidden = Dense(dense_hidden, activation='relu')(concat_noise)
	main_output = Dense(1, activation='sigmoid',name='output')(final_hidden)
	model = Model(inputs=[word_input], outputs=[main_output])
	model.compile(loss='mse',optimizer=update,metrics=['accuracy'])
	return model


def Graph_CNN_model(word_vocab_size, word_max_len, char_vocab_size, char_max_len, word_emb_dims, char_emb_dims, lstm_hidden_dims, nb_filter, filter_length, dense_hidden, word_pretrain_embedding,mask_zero,BILSTM, drop_prob,lstm_average, update="adam"):
	word_input = Input(shape=(word_max_len, ), dtype="int32", name='word_input')
	char_input = Input(shape=(char_max_len, ), dtype="int32", name='char_input')

	word_emb = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dims, input_length=word_max_len, embeddings_initializer=word_pretrain_embedding, mask_zero=False)(word_input)
	char_emb = Embedding(input_dim=char_vocab_size, output_dim=char_emb_dims, input_length=char_max_len, mask_zero=False)(char_input)

	word_drop = GaussianDropout(drop_prob)(word_emb)
	char_drop = GaussianDropout(drop_prob)(char_emb)
	
	## add cnn
	word_cnn = Conv1D(filters=nb_filter,kernel_size=filter_length,activation='relu')(word_drop)
	char_cnn = Conv1D(filters=nb_filter,kernel_size=filter_length,activation='relu')(char_drop)
	word_cnn_pool =  MaxPooling1D(pool_size=5)(word_cnn)
	char_cnn_pool =  MaxPooling1D(pool_size=5)(char_cnn)
	word_cnn_flat = Flatten()(word_cnn_pool)
	char_cnn_flat = Flatten()(char_cnn_pool)

	## concat all and dropout
	concat_lstm = concatenate([word_cnn_flat, char_cnn_flat])
	concat_noise = GaussianDropout(drop_prob)(concat_lstm)

	## final input
	final_hidden = Dense(dense_hidden, activation='relu')(concat_noise)
	main_output = Dense(1, activation='sigmoid',name='output')(final_hidden)
	model = Model(inputs=[word_input, char_input], outputs=[main_output])
	model.compile(loss='mse',optimizer=update,metrics=['accuracy'])
	return model

