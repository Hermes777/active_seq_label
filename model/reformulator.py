# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-07 17:09:34
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from charbilstm import CharBiLSTM
from charcnn import CharCNN
from torch.autograd import Variable

class Reformulator(nn.Module):
    def __init__(self, data):
        super(Reformulator, self).__init__()
        print "build batched bilstm..."
        self.gpu = data.HP_gpu
        self.use_char = data.HP_use_char
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        self.average_batch = data.HP_average_batch_loss
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if data.char_features == "CNN":
                self.char_feature = CharCNN(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_features == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            else:
                print "Error char feature selection, please check parameter data.char_features (either CNN or LSTM)."
                exit(0)
        self.embedding_dim = data.word_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.tag_size=data.label_alphabet_size
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim + self.char_hidden_dim+data.label_alphabet_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.hidden_dim, 2)
        self.topk = 50

        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.lstm = self.lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,tag_seq,tag_size):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs =  self.word_embeddings(word_inputs)

        if self.use_char:
            ## calculate char lstm last hidden
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            word_embs = torch.cat([word_embs, char_features], 2)

        tag_feature = torch.zeros(batch_size, sent_len, tag_size).cuda().scatter_(2,tag_seq.unsqueeze(2).cuda(),1.0)
        ## concat word and char together
        word_embs = self.drop(word_embs)
        word_embs = torch.cat([word_embs, Variable(tag_feature)], 2)
        packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        packed_words.requires_grad=False

        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1,0))
        ## lstm_out (batch_size, seq_len, hidden_size)
        return lstm_out


    def get_output_score(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, tag_seq):
        lstm_out = self.get_lstm_features(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, tag_seq,self.tag_size)
        outputs = self.hidden2tag(lstm_out)
        return outputs
    

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label,tag_seq, mask):
        ## mask is not used
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        #loss_function = nn.CrossEntropyLoss()
        outs = self.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,tag_seq)
        # outs (batch_size, seq_len, 2)
        outs = outs.view(total_word, -1)
        score = F.softmax(outs, 1)

        #loss = loss_function(score, batch_label.view(total_word))
        #if self.average_batch:
        #    loss = loss / batch_size
        #print('score0',loss)

        _=score[:,0]

        _=_.contiguous().view(batch_size,seq_len)
        #print(_)

        _=mask.float() * _#the score is always positive
        #print(mask)
        #print("_",_)
        _=F.softmax(_, 1)

        if seq_len>=self.topk:
            #print(self.topk)
            topk, indices = _.topk(self.topk,dim=1)
        else:
            #print(seq_len)
            topk, indices = _.topk(seq_len,dim=1)
        tag_mask=Variable(torch.ones(batch_size, seq_len).cuda())
        tag_mask=tag_mask.scatter(1,indices,0).long()
        #print("topk",topk)
        #topk=F.softmax(topk, 1)
        topk=torch.log(topk)
        #print(topk)

        #topk=torch.log(_)*(1-(-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())
        
        info_tensor=(1-(-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())#inequal if one
        _sum=info_tensor.sum().long()[0]
        #ans=torch.log(_)*(1-(-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())
        ans=0.0
        for i in range(_sum):
            ans+=torch.index_select(info_tensor[0],0,indices[0][:i+1]).sum()/float(i+1)

        #print(batch_label)
        #print(Variable(tag_seq).cuda())

        return indices,tag_mask, topk.mean(1),ans

        # tag_seq = autograd.Variable(torch.zeros(batch_size, seq_len)).long()
        # total_loss = 0
        # for idx in range(batch_size):
        #     score = F.log_softmax(outs[idx])
        #     loss = loss_function(score, batch_label[idx])
        #     # tag_seq[idx] = score.cpu().data.numpy().argmax(axis=1)
        #     _, tag_seq[idx] = torch.max(score, 1)
        #     total_loss += loss
        # return total_loss, tag_seq


    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask,tag_seq):
        
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        outs = self.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, tag_seq)
        outs = outs.view(total_word, -1)
        _, tag_seq  = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        ## filter padded position with zero
        decode_seq = mask.long() * tag_seq
        #print('outs',out)
        return decode_seq
        # # tag_seq = np.zeros((batch_size, seq_len), dtype=np.int)
        # tag_seq = autograd.Variable(torch.zeros(batch_size, seq_len)).long()
        # if self.gpu:
        #     tag_seq = tag_seq.cuda()
        # for idx in range(batch_size):
        #     score = F.log_softmax(outs[idx])
        #     _, tag_seq[idx] = torch.max(score, 1)
        #     # tag_seq[idx] = score.cpu().data.numpy().argmax(axis=1)
        # return tag_seq
