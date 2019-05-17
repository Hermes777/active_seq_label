# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from crf import CRF
from examiner import Examiner
from torch.autograd import Variable
import nltk
from sklearn_crfsuite import CRF
import math
 

class SeqModel():
    def __init__(self, data):

        print "build batched lstmcrf..."

        self.label_alphabet=data.label_alphabet
        self.word_alphabet=data.word_alphabet

        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=False
        )
        self.examiner = Examiner(data)
        self.useExaminer = False
        self.loss_function = nn.NLLLoss()
        self.topk=5
        self.X_train=[]
        self.Y_train=[]
        self.pos_mask_list=[]
        self.instances=[]
        self.scores_refs=[]
        self.pos_mask=None
        self.tag_size=data.label_alphabet_size

    #For the afterward updating of the crf
    def masked_label(self,pos_mask,mask,batch_label,tag_seq):


        batch_label=batch_label.mul(1-pos_mask)

        tag_seq=Variable(tag_seq).cuda().mul(pos_mask)

        return batch_label+tag_seq

    def ner(sentence):
        sentence_features = [self.features(sentence, index) for index in range(len(sentence))]
        return list(zip(sentence, model.predict([sentence_features])[0]))

    def rand_mask(self,word_inputs,mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        if self.full==True:
            return Variable(torch.zeros(batch_size,seq_len).cuda().long(),requires_grad=False)
        rand_vec=Variable(torch.rand(batch_size,seq_len),requires_grad=False)

        rand_vec=mask.float() * rand_vec.cuda()

        if seq_len>=self.topk:
            topk, indices = rand_vec.topk(self.topk,dim=1)
        else:
            topk, indices = rand_vec.topk(seq_len,dim=1)
        pos_mask=Variable(torch.ones(batch_size, seq_len).cuda())
        pos_mask=pos_mask.scatter(1,indices,0).long()
        return pos_mask
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
    
    def pos_selection(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask,t=None, pos_mask=None):
        
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        #get score and tag_seq
        #outs = self.lstm.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)

        tag_seq = self.sequence_to_tensor(self.label_alphabet,self.crf.predict(self.tensor_to_sequence(self.word_alphabet,word_inputs,label=False)))
        distributions=self.crf.predict_marginals(self.tensor_to_sequence(self.word_alphabet,word_inputs,label=False))

        tag_prob=Variable(torch.zeros(1,word_seq_lengths[0], self.tag_size).cuda())
        for j,key in enumerate(self.label_alphabet.instances):
            for i in range(word_seq_lengths[0]):
                if key in distributions[0][i]:
                    tag_prob[0,i,j]=distributions[0][i][key]
                else:
                    tag_prob[0,i,j]=0.0
        if t!=None:
            t_mask=self.pos_mask_list[t]

            indices, pos_mask, scores_ref,score,correct = self.examiner.neg_log_likelihood_loss(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,batch_label,tag_seq,tag_prob,mask*t_mask.byte())
        else:
            indices, pos_mask, scores_ref,score,correct = self.examiner.neg_log_likelihood_loss(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,batch_label,tag_seq,tag_prob,mask)
        #pos_mask = self.rand_mask(word_inputs,mask)#random mask
        self.pos_mask=pos_mask
        new_batch_label=self.masked_label(pos_mask,mask,batch_label, tag_seq)
        
        return new_batch_label,tag_seq,tag_prob,pos_mask,score,indices,scores_ref
    def add_instance(self,word_inputs,batch_label,pos_mask,instance,scores_ref):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        self.X_train.append(self.tensor_to_sequence(self.word_alphabet,word_inputs,label=False)[0])
        self.Y_train.append(self.tensor_to_sequence(self.label_alphabet,batch_label)[0])
        #print("self.tag_mask",self.tag_mask.size())

        if pos_mask is None:
            self.pos_mask_list.append(Variable(torch.zeros(batch_size, seq_len).long()).cuda()) 
        else:
            self.pos_mask_list.append(pos_mask) 
        self.instances.append(instance)
        self.scores_refs.append(scores_ref)
    def readd_instance(self,batch_label, mask,pos_mask, i,scores_ref):
        tag_seq = self.sequence_to_tensor(self.label_alphabet,self.crf.predict([self.X_train[i]]))

        pos_mask=self.pos_mask_list[i].long()*pos_mask.long()

        batch_label=self.masked_label(pos_mask,mask,batch_label, tag_seq)
        self.Y_train[i]=self.tensor_to_sequence(self.label_alphabet,batch_label)[0]
        self.pos_mask_list[i]=pos_mask
        self.scores_refs[i]=scores_ref


    def reinforment_reward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label,tag_seq,tag_prob, mask,mode):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
 
        indices,pos_mask,scores_ref,full_loss,partial_reward = self.examiner.neg_log_likelihood_loss(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,batch_label,tag_seq,tag_prob,mask)        
        '''
            indices: the selected positions as indices
            pos_mask: the selected positions as mask vector
            pos_mask: the selected positions as mask vector

        '''
        if mode=="supervised_partial":
            return pos_mask,(full_loss*(1-pos_mask.float())).sum()
        elif mode=="supervised_full":
            return pos_mask,full_loss
        else:
            return pos_mask,scores_ref



    def pop_instance(self,x):
        self.X_train.pop(0)
        self.Y_train.pop(0)
    def reevaluate_instance(self, mask):
        for i in range(len(self.X_train)):
            #X_train[i]
            tag_seq = self.sequence_to_tensor(self.label_alphabet,self.crf.predict([self.X_train[i]]))


            pos_mask=self.pos_mask_list[i]

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
            word1 = sent[i-1]
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
            word1 = sent[i+1]
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

        self.crf.fit(self.X_train, self.Y_train)
        return 
    def sample_train(self,left,right):

        self.crf.fit(self.X_train[left:right], self.Y_train[left:right])

        return 
    def test(self,word_inputs):
        tag_seq = self.sequence_to_tensor(self.label_alphabet,self.crf.predict(self.tensor_to_sequence(self.word_alphabet,word_inputs,label=False)))
        return Variable(tag_seq).cuda()



        