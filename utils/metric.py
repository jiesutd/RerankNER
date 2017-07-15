#!/usr/bin/env python
# coding=utf-8

##===================================================================##
#   Utils:load data
#   Jie Yang
#   Sep. 7, 2016
# 
##===================================================================##

# from operator import add

import numpy as np
import math
from data_processor import load_rerank_data


def get_golden_predict_choose_results(structured_sentences, predict_result):
    whole_index = 0
    right_num = 0
    golden_choose_list = []
    predict_choose_list = []
    for each_candidates in structured_sentences:
        candidate_index = 0
        predict_candidate_score = -1
        predict_opt_candidate = -1
        golden_candidate_score = -1
        golden_opt_candidate = -1
        for each_sentence in each_candidates:
            if (predict_result[whole_index] > predict_candidate_score):
                predict_candidate_score = predict_result[whole_index]
                predict_opt_candidate = candidate_index
            ## regard highest accuracy candidate as gold 
            if each_sentence['accuracy'] > golden_candidate_score:
                golden_candidate_score = each_sentence['accuracy']
                golden_opt_candidate = candidate_index
            candidate_index += 1
            whole_index += 1
        golden_choose_list.append(golden_opt_candidate)
        predict_choose_list.append(predict_opt_candidate)
    assert(len(predict_result)== whole_index)
    return golden_choose_list,predict_choose_list



def get_alpha_golden_predict_choose_results(structured_sentences, predict_result, alpha):
    whole_index = 0
    right_num = 0
    golden_choose_list = []
    predict_choose_list = []
    for each_candidates in structured_sentences:
        candidate_index = 0
        predict_candidate_score = -1
        predict_opt_candidate = -1
        golden_candidate_score = -1
        golden_opt_candidate = -1
        for each_sentence in each_candidates:
            predict_value = predict_result[whole_index] * alpha + (1-alpha)* each_sentence['prob']
            if (predict_value > predict_candidate_score):
                predict_candidate_score = predict_value
                predict_opt_candidate = candidate_index
             ## regard highest accuracy candidate as gold 
            if each_sentence['accuracy'] > golden_candidate_score:
                golden_candidate_score = each_sentence['accuracy']
                golden_opt_candidate = candidate_index
            candidate_index += 1
            whole_index += 1
        golden_choose_list.append(golden_opt_candidate)
        predict_choose_list.append(predict_opt_candidate)
    return golden_choose_list,predict_choose_list


def candidate_choose_accuracy(golden_list, predict_list):
    result_num = len(golden_list)
    same_number = 0
    assert (result_num == len(predict_list)),"Golden and predict result size not match!"
    for idx in range(result_num):
      if golden_list[idx] == predict_list[idx]:
          same_number += 1
    accuracy = (same_number+0.0)/result_num
    # print "Total instances: ", result_num, "; Correct choice: ", same_number, "; Accuracy: ",accuracy
    return accuracy


def get_rerank_ner_fmeasure(structured_sentences, predict_choose_list, label_type="BMES"):
    seq_num = len(predict_choose_list)
    assert(len(structured_sentences) == seq_num), "structured_sentence and predict choose num not match! sent_Num:predict"
    sentence_list = []
    golden_list = []
    predict_list = []
    for idx in range(0, seq_num):
        golden_list.append(structured_sentences[idx][predict_choose_list[idx]]['gold_label'])
        predict_list.append(structured_sentences[idx][predict_choose_list[idx]]['pred_label'])
    return get_ner_fmeasure(golden_list,predict_list, label_type)


def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = "Nan"
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = 'Nan'
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == "Nan") or (recall == "Nan") or (precision+recall) <= 0.0:
        f_measure = "Nan"
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    # print "gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num
    return precision, recall, f_measure


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string



def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)
            
        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-' 
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag 
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def oracle_best_accuracy(structured_instance, nbest, label_type="BMES"):
    instance_num = len(structured_instance)
    max_candidate_num = 0
    for instance in structured_instance:
        if len(instance) > max_candidate_num:
            max_candidate_num = len(instance)
    if nbest > max_candidate_num:
        print ("Nbest over large than data candidate num, %s > %s"%(nbest, max_candidate_num))
    right_choose_num = 0
    for instance in structured_instance:
        if len(instance) > nbest:
            end = nbest
        else:
            end = len(instance)
        True_flag = False
        for idx in range(0, end):
            # print instance[idx][-4]
            if instance[idx][-4] >= 1.0:
                True_flag = True
        if True_flag:
            right_choose_num += 1
    accuracy = (right_choose_num+0.0)/instance_num
    print ("N-best: %s, Right num: %s, Total num: %s, Accuracy: %s."%(nbest,right_choose_num,instance_num,accuracy))
    return accuracy


def oracle_best_f(structured_instance, nbest, label_type="BMES"):
    instance_num = len(structured_instance)
    max_candidate_num = 0

    for instance in structured_instance:
        if len(instance) > max_candidate_num:
            max_candidate_num = len(instance)
    if nbest > max_candidate_num:
        print ("Nbest over large than data candidate num, %s > %s"%(nbest, max_candidate_num))
    predict_choose_list = []
    for instance in structured_instance:
        if len(instance) > nbest:
            end = nbest
        else:
            end = len(instance)
        True_flag = False
        best_pos = 0
        best_acc = -1
        for idx in range(0, end):
            if instance[idx]['target'] > best_acc:
                best_acc = instance[idx]['target']
                best_pos = idx
        predict_choose_list.append(best_pos)
    p,r,f = get_rerank_ner_fmeasure(structured_instance,predict_choose_list, label_type)
    # print "instance num:", instance_num
    # print ("P:%s, R:%s, F:%s"%(p,r,f))
    return f


def oracle_worst_f(structured_instance, nbest):
    instance_num = len(structured_instance)
    max_candidate_num = 0

    for instance in structured_instance:
        if len(instance) > max_candidate_num:
            max_candidate_num = len(instance)
    if nbest > max_candidate_num:
        print ("Nbest over large than data candidate num, %s > %s"%(nbest, max_candidate_num))
    predict_choose_list = []
    for instance in structured_instance:
        if len(instance) > nbest:
            end = nbest
        else:
            end = len(instance)
        True_flag = False
        worst_pos = 0
        worst_acc = 2
        for idx in range(0, end):
            if instance[idx]['target'] < worst_acc:
                worst_acc = instance[idx]['target']
                worst_pos = idx
        predict_choose_list.append(worst_pos)
    p,r,f = get_rerank_ner_fmeasure(structured_instance,predict_choose_list, label_type)
    print ("P:%s, R:%s, F:%s"%(p,r,f))
    return f



def accuracy_with_word_length(structured_instance, nbest):
    instance_num = len(structured_instance)
    max_candidate_num = 0
    length_dict = {}
    right_length_dict = {}
    for instance in structured_instance:
        if len(instance) > max_candidate_num:
            max_candidate_num = len(instance)
    if nbest > max_candidate_num:
        print ("Nbest over large than data candidate num, %s > %s"%(nbest, max_candidate_num))
    right_choose_num = 0
    for instance in structured_instance:
        word_length = len(instance[0]['orig_word'])
        if len(instance) > nbest:
            end = nbest
        else:
            end = len(instance)
        True_flag = False
        for idx in range(0, end):
            if instance[idx][-4] >= 1.0:
                True_flag = True
        if True_flag:
            if word_length in right_length_dict:
                right_length_dict[word_length] += 1
            else:
                right_length_dict[word_length] = 1
        if word_length in length_dict:
            length_dict[word_length] += 1
        else:
            length_dict[word_length] = 1  
    for each_key in length_dict.keys():
        if each_key not in right_length_dict:
            right_length_dict[each_key] = 0
    cluster_length_dict = {}
    cluster_right_length_dict = {}
    for each_key in length_dict.keys():
        if  0< each_key <6:
            new_tag = 1
            if new_tag not in cluster_length_dict:
                cluster_length_dict[new_tag] = 0
            cluster_length_dict[new_tag] += length_dict[each_key]
            if new_tag not in cluster_right_length_dict:
                cluster_right_length_dict[new_tag] = 0
            cluster_right_length_dict[new_tag] += right_length_dict[each_key]
        elif each_key < 11:
            new_tag = 2
            if new_tag not in cluster_length_dict:
                cluster_length_dict[new_tag] = 0
            cluster_length_dict[new_tag] += length_dict[each_key]
            if new_tag not in cluster_right_length_dict:
                cluster_right_length_dict[new_tag] = 0
            cluster_right_length_dict[new_tag] += right_length_dict[each_key]
        elif each_key < 16:
            new_tag = 3
            if new_tag not in cluster_length_dict:
                cluster_length_dict[new_tag] = 0
            cluster_length_dict[new_tag] += length_dict[each_key]
            if new_tag not in cluster_right_length_dict:
                cluster_right_length_dict[new_tag] = 0
            cluster_right_length_dict[new_tag] += right_length_dict[each_key]
        elif each_key < 21:
            new_tag = 4
            if new_tag not in cluster_length_dict:
                cluster_length_dict[new_tag] = 0
            cluster_length_dict[new_tag] += length_dict[each_key]
            if new_tag not in cluster_right_length_dict:
                cluster_right_length_dict[new_tag] = 0
            cluster_right_length_dict[new_tag] += right_length_dict[each_key]
        elif each_key < 26:
            new_tag = 5
            if new_tag not in cluster_length_dict:
                cluster_length_dict[new_tag] = 0
            cluster_length_dict[new_tag] += length_dict[each_key]
            if new_tag not in cluster_right_length_dict:
                cluster_right_length_dict[new_tag] = 0
            cluster_right_length_dict[new_tag] += right_length_dict[each_key]
        elif each_key < 31:
            new_tag = 6
            if new_tag not in cluster_length_dict:
                cluster_length_dict[new_tag] = 0
            cluster_length_dict[new_tag] += length_dict[each_key]
            if new_tag not in cluster_right_length_dict:
                cluster_right_length_dict[new_tag] = 0
            cluster_right_length_dict[new_tag] += right_length_dict[each_key]
        elif each_key < 41:
            new_tag = 7
            if new_tag not in cluster_length_dict:
                cluster_length_dict[new_tag] = 0
            cluster_length_dict[new_tag] += length_dict[each_key]
            if new_tag not in cluster_right_length_dict:
                cluster_right_length_dict[new_tag] = 0
            cluster_right_length_dict[new_tag] += right_length_dict[each_key]
        elif each_key < 51:
            new_tag = 8
            if new_tag not in cluster_length_dict:
                cluster_length_dict[new_tag] = 0
            cluster_length_dict[new_tag] += length_dict[each_key]
            if new_tag not in cluster_right_length_dict:
                cluster_right_length_dict[new_tag] = 0
            cluster_right_length_dict[new_tag] += right_length_dict[each_key]
        # elif each_key < 61:
        #     new_tag = 9
        #     if new_tag not in cluster_length_dict:
        #         cluster_length_dict[new_tag] = 0
        #     cluster_length_dict[new_tag] += length_dict[each_key]
        #     if new_tag not in cluster_right_length_dict:
        #         cluster_right_length_dict[new_tag] = 0
        #     cluster_right_length_dict[new_tag] += right_length_dict[each_key]
        # elif each_key < 71:
        #     new_tag = 10
        #     if new_tag not in cluster_length_dict:
        #         cluster_length_dict[new_tag] = 0
        #     cluster_length_dict[new_tag] += length_dict[each_key]
        #     if new_tag not in cluster_right_length_dict:
        #         cluster_right_length_dict[new_tag] = 0
        #     cluster_right_length_dict[new_tag] += right_length_dict[each_key]
        else:
            new_tag = 9
            if new_tag not in cluster_length_dict:
                cluster_length_dict[new_tag] = 0
            cluster_length_dict[new_tag] += length_dict[each_key]
            if new_tag not in cluster_right_length_dict:
                cluster_right_length_dict[new_tag] = 0
            cluster_right_length_dict[new_tag] += right_length_dict[each_key]



    accuracy_dict = {}
    for each_key in cluster_length_dict.keys():
        accuracy_dict[each_key] = (cluster_right_length_dict[each_key] + 0.0)/cluster_length_dict[each_key]
    length_list = []
    accuracy_list = []
    for w in sorted(accuracy_dict):
        length_list.append(w)
        accuracy_list.append(accuracy_dict[w])
    # print accuracy_dict
    return length_list, accuracy_list
            
    


def save_predict_result(structured_sentences, predict_choose_list, output_file):
    out_file = open(output_file,'w')
    seq_num = len(predict_choose_list)
    assert(len(structured_sentences) == seq_num), "structured_sentence and predict choose num not match! sent_Num:predict"
    for idx in range(0, seq_num):
        sentence_list = structured_sentences[idx][predict_choose_list[idx]]['orig_word']
        golden_list= structured_sentences[idx][predict_choose_list[idx]]['gold_label']
        predict_list= structured_sentences[idx][predict_choose_list[idx]]['pred_label']
        sent_length = len(sentence_list)
        assert(sent_length== len(predict_list))
        for idy in range(sent_length):
            out_file.write(sentence_list[idy]+" "+ golden_list[idy]+ " "+ predict_list[idy]+ '\n')
        out_file.write('\n')
    print "Result has been written in", output_file
    print "Sentence num:", seq_num



if __name__ == '__main__':
    debug = '../data/small_dev.txt'
    input_file = '../data/test.bio.filter'
    structure_sent = load_rerank_data(input_file)
    print oracle_best_f(structure_sent, 1, 'BIO')






