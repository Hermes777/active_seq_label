# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import string

# The class for an Examiner
class Examiner(nn.Module):
    def __init__(self, data):
        super(Examiner, self).__init__()
        print("build batched bilstm...")
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
                print("Error char feature selection, please check parameter data.char_features (either CNN or LSTM).")
                exit(0)
        self.embedding_dim = data.word_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.label_alphabet=data.label_alphabet.instances
        self.tag_size=len(self.label_alphabet)

        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim + self.char_hidden_dim+data.label_alphabet_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

        self.feature_dim=368
        self.feature_hid=100
        self.feature_emb=30

        self.hidden2tag = nn.Linear(self.hidden_dim+self.feature_emb, 2)
        self.topk = 50

        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.lstm = self.lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()

        self.channel=['+1:','-1:','']
        self.word=['word:'+x for x in list(string.ascii_lowercase)]
        self.word.append('word:NUM')
        self.word.append('part0')
        self.word.append('part1')
        self.word.append('part2')
        self.word.append('part3')
        self.word=sorted(self.word)
        self.channel=sorted(self.channel)
        self.tag_size=data.label_alphabet_size
        self.conv1 = torch.nn.Conv2d(3,4,kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(4,8,kernel_size=(5,3), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8,8,kernel_size=(7,5), stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.feature2hid=nn.Linear(self.feature_dim, self.feature_hid)    
        self.hid2emb=nn.Linear(self.feature_hid, self.feature_emb)     
   
    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,tag_prob,tag_size,crf):
        """
            input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
                crf: the crf component of the model
            construct the neural network
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs =  self.word_embeddings(word_inputs)

        if self.use_char:

            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            word_embs = torch.cat([word_embs, char_features], 2)

        tag_feature = tag_prob.cuda()
        word_embs = self.drop(word_embs)
        word_embs = torch.cat([word_embs, tag_feature], 2)
        packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        packed_words.requires_grad=False

        lstm_out, hidden = self.lstm(packed_words, hidden)

        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1,0))

        feature_in=torch.zeros(1,3,self.tag_size,32).float()
        for i1,key1 in enumerate(self.channel):
            for i2,key2 in enumerate(self.word):
                for i3,key3 in enumerate(self.label_alphabet):
                    key=(key1+key2,key3)
                    if key in crf.state_features_:
                        feature_in[0,i1,i3,i2]=crf.state_features_[key]

        feature_in=F.relu(self.conv1(feature_in))
        feature_in=self.pool(feature_in)
        feature_in=F.relu(self.conv2(feature_in))
        feature_in=self.pool(feature_in)
        feature_in=F.relu(self.conv3(feature_in))
        crf_feature=feature_in.view(1,-1)



        

        crf_feature=F.relu(self.feature2hid(Variable(crf_feature)))
        crf_feature=self.hid2emb(Variable(crf_feature))

        crf_feature=crf_feature.expand(1,sent_len,-1).cuda()

        lstm_out = torch.cat([lstm_out, crf_feature], 2)

        return lstm_out


    def get_output_score(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, tag_seq,crf):
        """
            input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
                crf: the crf component of the model
            output:
                outputs as a vector
        """
        lstm_out = self.get_lstm_features(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, tag_seq,self.tag_size,crf)
        outputs = self.hidden2tag(lstm_out)
        return outputs
    
    def sample(self,dist):
        #which might be useful for smapling case
        choice = torch.multinomial(dist, num_samples=1, replacement=True)
        return choice

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label,tag_seq,tag_prob, mask,crf):
        """
            input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
                crf: the crf component of the model
            output:
                indices: the indices that have been chosen
                tag_mask: 0/1 vextor as a mask, 0 means been chosen
                prob_max: the max probability as a variable which can be back-probgated
                full_loss: full supervised loss
        """
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len

        outs= self.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,tag_prob,crf)
        # outs (batch_size, seq_len, 2)
        outs = outs.view(total_word, -1)
        score = F.softmax(outs, 1)
        # score: The score for choosing each position

        score=score[:,0]

        score=score.contiguous().view(batch_size,seq_len)

        score=mask.float() * score#This step is only useful for NER case

        score=F.softmax(score, 1)
        score_multi=Variable(torch.Tensor(5,seq_len-4).cuda())

        score_multi=torch.cat([score[:,:-4],score[:,1:-3],score[:,2:-2],score[:,3:-1],score[:,4:]],dim=0)

        score_multi=torch.log(score_multi)

        score_multi=torch.sum(score_multi,0)

        score_multi=torch.exp(score_multi)
        prob_max,indices_max=torch.max(score_multi,dim=0)

        if self.gpu:
            indices=torch.tensor([[indices_max,indices_max+1,indices_max+2,indices_max+3,indices_max+4]]).cuda()
            tag_mask=Variable(torch.ones(batch_size, seq_len).cuda())
        else:
            indices=torch.tensor([[indices_max,indices_max+1,indices_max+2,indices_max+3,indices_max+4]])
            tag_mask=Variable(torch.ones(batch_size, seq_len))

        tag_mask=tag_mask.scatter(1,indices,0).long().cuda()


        prob_max = (torch.log(prob_max)-torch.log(torch.sum(score_multi)))

        
        if self.gpu:
            info_tensor=(1-(-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())#inequal if one
        else:
            info_tensor=(1-(-torch.abs(Variable(tag_seq)-batch_label)).ge(0).float())
        #print(info_tensor.sum().long())
        _sum=info_tensor.sum().long()

        if self.gpu:
            full_loss=-torch.log(score)*(1-(-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())
            partial_reward=score*(1-(-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())
        else:
            full_loss=-torch.log(score)*(1-(-torch.abs(Variable(tag_seq)-batch_label)).ge(0).float())
            partial_reward=score*(1-(-torch.abs(Variable(tag_seq)-batch_label)).ge(0).float())



        return indices,tag_mask, prob_max,full_loss





