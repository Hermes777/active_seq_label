# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2017-12-14 12:04:48

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bilstm import BiLSTM
#from crf import CRF
from reformulator import Reformulator
from torch.autograd import Variable
import nltk
from sklearn_crfsuite import CRF
import math
 

class BiLSTM_CRF():
    def __init__(self, data):
        #super(BiLSTM_CRF, self).__init__()
        print "build batched lstmcrf..."

        ## add two more label for downlayer lstm, use original label size for CRF
        #label_size = data.label_alphabet_size
        self.label_alphabet=data.label_alphabet
        self.word_alphabet=data.word_alphabet
        #self.label_alphabet_size += 2
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.reformulator = Reformulator(data)
        self.useReformulator = False
        self.loss_function = nn.NLLLoss()
        self.topk=50
        self.X_train=[]
        self.Y_train=[]
        self.tag_mask_list=[]
        self.instances=[]
        self.scores_refs=[]
        self.tag_mask=None

    #For the afterward updating of the crf
    def masked_label(self,pos_mask,mask,batch_label,tag_seq):


        #batch_label=batch_label.mul(1-pos_mask)

        #tag_seq=Variable(tag_seq).cuda().mul(pos_mask)

        return batch_label#+tag_seq

    def ner(sentence):
        sentence_features = [self.features(sentence, index) for index in range(len(sentence))]
        return list(zip(sentence, model.predict([sentence_features])[0]))

    def rand_mask(self,word_inputs,mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        if self.topk==50:
            return Variable(torch.zeros(batch_size,seq_len).cuda().long(),requires_grad=False)
        _=Variable(torch.rand(batch_size,seq_len),requires_grad=False)

        _=mask.float() * _.cuda()

        if seq_len>=self.topk:
            topk, indices = _.topk(self.topk,dim=1)
        else:
            topk, indices = _.topk(seq_len,dim=1)
        tag_mask=Variable(torch.ones(batch_size, seq_len).cuda())
        tag_mask=tag_mask.scatter(1,indices,0).long()
        print("tag_mask",tag_mask)
        return tag_mask
    #For the afterward updating of the crf
    def sent2features(self,sent):
        return [self.features(sent, i) for i in range(len(sent))]
    def sent2labels(sent):
        return [label for token, postag, label in sent]
    def sent2tokens(sent):
        return [token for token, postag, label in sent]

    def tensor_to_sequence(self, _alphabet, word_inputs, label=True):
        #seq_len = word_inputs.size(1)
        if label==True:
            return [[_alphabet.get_instance(x.data[0]) for x in word_inputs[0]]]
        else:
            return [self.sent2features([_alphabet.get_instance(x.data[0]) for x in word_inputs[0]])]
    def sequence_to_tensor(self,_alphabet, word_inputs):
        return torch.LongTensor([[_alphabet.get_index(x) for x in word_inputs[0]]])
    
    def crf_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask,t=None):
        
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        #get score and tag_seq
        #outs = self.lstm.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)

        tag_seq = self.sequence_to_tensor(self.label_alphabet,self.crf.predict(self.tensor_to_sequence(self.word_alphabet,word_inputs,label=False)))
        if t!=None:
            t_mask=self.tag_mask_list[t]
            #print("t_mask",t_mask)
            indices, pos_mask, scores_ref,score = self.reformulator.neg_log_likelihood_loss(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,batch_label,tag_seq,mask*t_mask.byte())
        else:
            indices, pos_mask, scores_ref,score = self.reformulator.neg_log_likelihood_loss(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,batch_label,tag_seq,mask)
        pos_mask = self.rand_mask(word_inputs,mask)#currently we are using random mask
        self.tag_mask=pos_mask


        batch_label=self.masked_label(pos_mask,mask,batch_label, tag_seq)

        #total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)

        
        return batch_label,tag_seq,pos_mask,score,indices,scores_ref
    def add_instance(self,word_inputs,batch_label,tag_mask,instance,scores_ref):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        self.X_train.append(self.tensor_to_sequence(self.word_alphabet,word_inputs,label=False)[0])
        self.Y_train.append(self.tensor_to_sequence(self.label_alphabet,batch_label)[0])
        #print("self.tag_mask",self.tag_mask.size())
        #print("self.Y_train",self.tensor_to_sequence(self.label_alphabet,batch_label)[0])
        if tag_mask is None:
            self.tag_mask_list.append(Variable(torch.zeros(batch_size, seq_len).long()).cuda()) 
        else:
            self.tag_mask_list.append(tag_mask) 
        self.instances.append(instance)
        self.scores_refs.append(scores_ref)
    def readd_instance(self,batch_label, mask,tag_mask, i,scores_ref):
        tag_seq = self.sequence_to_tensor(self.label_alphabet,self.crf.predict([self.X_train[i]]))

        pos_mask=self.tag_mask_list[i].long()*tag_mask.long()

        batch_label=self.masked_label(pos_mask,mask,batch_label, tag_seq)
        self.Y_train[i]=self.tensor_to_sequence(self.label_alphabet,batch_label)[0]
        self.tag_mask_list[i]=pos_mask
        self.scores_refs[i]=scores_ref


    def reinforment_reward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label,tag_seq, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        #get score and tag_seq
        #outs = self.lstm.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        #tag_seq = self.sequence_to_tensor(self.label_alphabet,self.crf.predict(self.tensor_to_sequence(self.word_alphabet,word_inputs,label=False)))
        #print("tag_seq",score)
        #print(batch_label)

        #get_selected position
        indices,pos_mask,scores_ref,score = self.reformulator.neg_log_likelihood_loss(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,batch_label,tag_seq,mask)
        #if self.average_batch:
        #    total_loss = total_loss / batch_size
        #    total_loss1 = total_loss1 / batch_size
        

        return pos_mask,scores_ref

    def pop_instance(self,x):
        self.X_train.pop(0)
        self.Y_train.pop(0)
    def reevaluate_instance(self, mask):
        for i in range(len(self.X_train)):
            #X_train[i]
            tag_seq = self.sequence_to_tensor(self.label_alphabet,self.crf.predict([self.X_train[i]]))


            pos_mask=self.tag_mask_list[i]

            batch_label=self.masked_label(pos_mask,mask,Variable(self.sequence_to_tensor(self.label_alphabet,[self.Y_train[i]])).cuda(), tag_seq)
            self.Y_train[i]=self.tensor_to_sequence(self.label_alphabet,batch_label)[0]


    def features(self,sent, i):
        # obtain some overall information of the point name string
        num_part = 4
        len_string = len(sent)
        mod = len_string % num_part
        part_size = int(math.floor(len_string/num_part))
        # determine which part the current character belongs to
        # larger part will be at the beginning if the whole sequence can't be divided evenly
        size_list = []
        mod_count = 0
        for j in range(num_part):
            if mod_count < mod:
                size_list.append(part_size+1)
                mod_count += 1
            else:
                size_list.append(part_size)
        # for current character
        part_cumulative = [0]*num_part
        for j in range(num_part):
            if j > 0:
                part_cumulative[j] = part_cumulative[j-1] + size_list[j]
            else:
                part_cumulative[j] = size_list[j] - 1   # indices start from 0
        part_indicator = [0]*num_part
        for j in range(num_part):
            if part_cumulative[j] >= i:
                part_indicator[j] = 1
                break
        word = sent[i][0]
        if word.isdigit():
            itself = 'NUM'
        else:
            itself = word
        features = {
            'word': itself,
            'part0': part_indicator[0] == 1,
            'part1': part_indicator[1] == 1,
            'part2': part_indicator[2] == 1,
            'part3': part_indicator[3] == 1,
        }
        # for previous character
        if i > 0:
            part_indicator = [0] * num_part
            for j in range(num_part):
                if part_cumulative[j] >= i-1:
                    part_indicator[j] = 1
                    break
            word1 = sent[i-1][0]
            if word1.isdigit():
                itself1 = 'NUM'
            else:
                itself1 = word1
            features.update({
                '-1:word': itself1,
                '-1:part0': part_indicator[0] == 1,
                '-1:part1': part_indicator[1] == 1,
                '-1:part2': part_indicator[2] == 1,
                '-1:part3': part_indicator[3] == 1,
            })
        else:
            features['BOS'] = True
        # for next character
        if i < len(sent)-1:
            part_indicator = [0] * num_part
            for j in range(num_part):
                if part_cumulative[j] >= i + 1:
                    part_indicator[j] = 1
                    break
            word1 = sent[i+1][0]
            if word1.isdigit():
                itself1 = 'NUM'
            else:
                itself1 = word1
            features.update({
                '+1:word': itself1,
                '+1:part0': part_indicator[0] == 1,
                '+1:part1': part_indicator[1] == 1,
                '+1:part2': part_indicator[2] == 1,
                '+1:part3': part_indicator[3] == 1,
            })
        else:
            features['EOS'] = True
        return features
        #
    
    def train(self):
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

        self.crf.fit(self.X_train, self.Y_train)
        return 
    def test(self,word_inputs):
        tag_seq = self.sequence_to_tensor(self.label_alphabet,self.crf.predict(self.tensor_to_sequence(self.word_alphabet,word_inputs,label=False)))
        return Variable(tag_seq).cuda()



        