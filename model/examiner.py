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
        # The linear layer that maps from hidden state space to tag space
        self.feature_dim=368
        self.feature_hid=100
        self.feature_emb=30
        #self.feature2emb=nn.Linear(self.feature_dim, self.feature_emb)

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
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs =  self.word_embeddings(word_inputs)
        #print "word_embs",word_embs
        if self.use_char:
            ## calculate char lstm last hidden
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            word_embs = torch.cat([word_embs, char_features], 2)
        #print(tag_prob)
        tag_feature = tag_prob.cuda()#torch.zeros(batch_size, sent_len, tag_size).cuda().scatter_(2,tag_seq.unsqueeze(2).cuda(),1.0)
        word_embs = self.drop(word_embs)
        word_embs = torch.cat([word_embs, tag_feature], 2)
        packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        packed_words.requires_grad=False

        lstm_out, hidden = self.lstm(packed_words, hidden)

        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1,0))

        #print "out",lstm_out
        #print(len(sorted(crf.transition_features_.keys())))
        #crf_feature=torch.tensor([])
        feature_in=torch.zeros(1,3,self.tag_size,32).float()
        #print(self.label_alphabet)
        for i1,key1 in enumerate(self.channel):
            for i2,key2 in enumerate(self.word):
                for i3,key3 in enumerate(self.label_alphabet):
                    key=(key1+key2,key3)
                    if key in crf.state_features_:
                        feature_in[0,i1,i3,i2]=crf.state_features_[key]
        #print(feature_in.size())
        feature_in=F.relu(self.conv1(feature_in))
        feature_in=self.pool(feature_in)
        feature_in=F.relu(self.conv2(feature_in))
        feature_in=self.pool(feature_in)
        feature_in=F.relu(self.conv3(feature_in))
        crf_feature=feature_in.view(1,-1)

        #print("size",crf_feature.size())

        
        #crf_feature=torch.FloatTensor(np.array([[[self.feature_in[key] for key in sorted(self.feature_set)]]]))
        crf_feature=F.relu(self.feature2hid(Variable(crf_feature)))
        crf_feature=self.hid2emb(Variable(crf_feature))
        #crf_feature=Variable(torch.Tensor(zeros(1,30)))
        #print "crf",crf_feature
        crf_feature=crf_feature.expand(1,sent_len,-1).cuda()
        #print "crf1",crf_feature
        #print(lstm_out)
        #print(crf_feature)
        lstm_out = torch.cat([lstm_out, crf_feature], 2)
        ## lstm_out (batch_size, seq_len, hidden_size)
        return lstm_out


    def get_output_score(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, tag_seq,crf):
        lstm_out = self.get_lstm_features(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, tag_seq,self.tag_size,crf)
        outputs = self.hidden2tag(lstm_out)
        #print(outputs)
        return outputs,outputs
    
    def sample(self,dist):
        # dist is a tensor of shape (batch_size  x vocab_size)
        choice = torch.multinomial(dist, num_samples=1, replacement=True)
        #choice = choice.squeeze(1).view(*dist.size()[:2])
        return choice

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label,tag_seq,tag_prob, mask,crf):

        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len

        outs,outs1 = self.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,tag_prob,crf)
        # outs (batch_size, seq_len, 2)
        outs = outs.view(total_word, -1)
        score = F.softmax(outs, 1)
        score1 = score
        # score: The score for choosing each position

        score=score[:,0]

        score=score.contiguous().view(batch_size,seq_len)

        score=mask.float() * score#the score is always positive
        # with torch.no_grad():
        #     score_norm=torch.norm(score,2)
        # score=score.div_(score_norm)
        score=F.softmax(score, 1)
        score_multi=Variable(torch.Tensor(5,seq_len-4).cuda())
        # print("score_multi",score_multi)
        # print("score",score[0][:-4])
        score_multi=torch.cat([score[:,:-4],score[:,1:-3],score[:,2:-2],score[:,3:-1],score[:,4:]],dim=0)
        # print("score",score_multi)
        score_multi=torch.log(score_multi)
        #print("score1",score_multi)
        score_multi=torch.sum(score_multi,0)
        #print("score2",score_multi)
        score_multi=torch.exp(score_multi)
        #print("score3",score_multi)
        #print("asdfasd",score)

        # if seq_len>=self.topk:
        #     topk, indices = score.topk(self.topk,dim=1)
        # else:
        #     topk, indices = score.topk(seq_len,dim=1)
        topk,indices_max=torch.max(score_multi,dim=0)
        # print("sc",indices)
        if self.gpu:
            indices=torch.tensor([[indices_max,indices_max+1,indices_max+2,indices_max+3,indices_max+4]]).cuda()
            tag_mask=Variable(torch.ones(batch_size, seq_len).cuda())
        else:
            indices=torch.tensor([[indices_max,indices_max+1,indices_max+2,indices_max+3,indices_max+4]])
            tag_mask=Variable(torch.ones(batch_size, seq_len))
        #indices=self.sample(score)
        # print("indices")
        # print(indices)
        # print("topk")
        # print(topk)
        tag_mask=tag_mask.scatter(1,indices,0).long().cuda()
        # tag mask: selected positons as mask vector 
        #print("tag_mask",indices)


        topk_grad=torch.tensor([-1/torch.sum(score_multi)]*10)
        topk_grad[indices_max]=topk_grad[indices_max]+1/topk
        topk = (torch.log(topk)-torch.log(torch.sum(score_multi)))

        #tag_mask = tag_mask.()

        #topk: the topk scores
        
        if self.gpu:
            info_tensor=(1-(-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())#inequal if one
        else:
            info_tensor=(1-(-torch.abs(Variable(tag_seq)-batch_label)).ge(0).float())
        #print(info_tensor.sum().long())
        _sum=info_tensor.sum().long()

        if self.gpu:
            full_loss=-torch.log(score)*(1-(-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())
            #print("ge")
            #print((1-(-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float()))
            #full_loss+=torch.log(score)*((-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())
            partial_reward=score*(1-(-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())
        else:
            full_loss=-torch.log(score)*(1-(-torch.abs(Variable(tag_seq)-batch_label)).ge(0).float())
            #full_loss+=torch.log(score)*((-torch.abs(Variable(tag_seq).cuda()-batch_label)).ge(0).float())
            partial_reward=score*(1-(-torch.abs(Variable(tag_seq)-batch_label)).ge(0).float())
        #full_loss: the supervised loss
        #partial_loss: the partial labeled supervised reward
        #print("topk",topk)

        return indices,tag_mask, topk,full_loss,topk_grad,score_multi
    def forward(self,score,topk_grad, reward):
        print("score",score)
        print("topk_grad",topk_grad)
        score.backward(reward*topk_grad.cuda())
        #loss.backward()




