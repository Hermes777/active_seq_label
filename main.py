# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-19 11:36:53

import time
import sys
import argparse
import random
import copy
import torch
import gc
import cPickle as pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from utils.metric import get_ner_fmeasure
from model.bilstmcrf import BiLSTM_CRF as SeqModel
from utils.data import Data

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    #data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.fix_alphabet()


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "p:",pred, pred_tag.tolist()
        # print "g:", gold, gold_tag.tolist()
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    ## save data settings
    with open(save_file, 'w') as fp:
        pickle.dump(new_data, fp)
    print "Data setting saved to file: ", save_file


def load_data_setting(save_file):
    with open(save_file, 'r') as fp:
        data = pickle.load(fp)
    print "Data setting loaded from file: ", save_file
    data.show_data_summary()
    return data

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print " Learning rate is setted as:", lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate(data, model, name, qleft=None,qright=None,batch_size=1):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print "Error: wrong evaluate name,", name
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.reformulator.eval()
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    if qleft==None:
        qleft=0
        qright=total_batch
    for batch_id in range(qleft,qright):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end >train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model.test(batch_word)#porblem: why it contains zero
        # print "tag:",tag_seq
        # print "batch_label:",batch_label
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)

        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results  


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    chars = [sent[1] for sent in input_batch_list]
    labels = [sent[2] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]

    length_list = [map(len, pad_char) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask

def train(data, save_model_dir, seg=True):
    print "Training model..."
    data.show_data_summary()
    save_data_name = save_model_dir +".dset"
    save_data_setting(data, save_data_name)
    loss_function = nn.NLLLoss()
    model = SeqModel(data)
    #model=copy.deepcopy(premodel)
    optimizer = optim.SGD(model.reformulator.parameters(), lr=data.HP_lr, momentum=data.HP_momentum)
    best_dev = -1
    data.HP_iteration = 5
    USE_CRF=True
    ## start training
    acc_list=[]
    p_list=[]
    r_list=[]
    f_list=[]
    map_list=[]
    epoch_start = time.time()
    temp_start = epoch_start
    instance_count = 0

    #random.shuffle(data.train_Ids)
    ## set model in train model
    model.reformulator.train()
    model.reformulator.zero_grad()
    model.topk=5
    model.reformulator.topk=5
    batch_size = data.HP_batch_size
    batch_id = 0
    train_num = len(data.train_Ids)
    total_batch = train_num//batch_size+1
    gamma=0
    cnt=0
    click=0
    sum_click=0
    sum_p=0.0
    #if idx==0:
    #    selected_data=[batch_id for batch_id in range(0,total_batch//1000)]
    tag_mask=None
    max_Iter=100
    pretrain_Iter=80
    for batch_id in range(0,max_Iter):

        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end >train_num:
            end = train_num
        instance = data.train_Ids[start:end]
        if not instance:
            continue

        update_once=False

        start_time = time.time()
        #selected_data.append(batch_id)

        if batch_id>=pretrain_Iter:
            t=np.random.randint(0,len(model.X_train))

            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
            real_batch_label=batch_label
            model.train()
            batch_label,tag_seq,tag_mask,score,indices,scores_ref=model.crf_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
            model.add_instance(batch_word,batch_label,tag_mask,instance,scores_ref.data[0])
            print("len",len(model.Y_train))


            model.train()
            end_time = time.time()
            if click+5>=10:
                print("time",end_time-start_time)
        else:
            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
            model.add_instance(batch_word,batch_label,tag_mask,instance,-100000.0)
            print("label",batch_label)
            print("len",len(model.Y_train))
        
        #print(batch_wordlen)
        if batch_id<pretrain_Iter-1:
            continue

        click+=5
        sum_click+=5


        if batch_id==pretrain_Iter-1 or click>=10: # evaluate every 2 instances
            if batch_id==pretrain_Iter-1 :
                p_list.append([x for x in model.Y_train])
            model.train()
            speed, acc, p, r, f, _ = evaluate(data, model, "test")
            print(sum_click)
            print("Accuracy",acc)
            acc_list.append(acc)

            click-=10

        
    train_finish = time.time()
    speed, acc, p, r, f, _ = evaluate(data, model, "test")
    test_finish = time.time()
    test_cost = test_finish - train_finish
    if seg:
        print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
    else:
        print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))
    gc.collect() 
    file_dump=open("random_list.pkl","w")
    pickle.dump([acc_list,p_list,r_list,f_list,map_list],file_dump)
    file_dump.close()

    #print("Best Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(best_[0], best_[1], best_[2], best_[3], best_[4], best_[5]))



def load_model_decode(model_dir, data, name, gpu, seg=True):
    data.HP_gpu = gpu
    print "Load Model from file: ", model_dir
    model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    if not gpu:
        model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
        # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    else:
        model.load_state_dict(torch.load(model_dir))
        # model = torch.load(model_dir)
    
    print("Decode %s data ..."%(name))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CRF')
    parser.add_argument('--wordemb',  help='Embedding for words', default='glove')
    parser.add_argument('--charemb',  help='Embedding for chars', default='None')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/save.dset")
    parser.add_argument('--train', default="data/train.bmes") 
    parser.add_argument('--dev', default="data/dev.bmes" )  
    parser.add_argument('--test', default="data/test.bmes") 
    parser.add_argument('--seg', default="False") 
    parser.add_argument('--extendalphabet', default="True") 
    parser.add_argument('--raw') 
    parser.add_argument('--loadmodel')
    parser.add_argument('--output') 
    args = parser.parse_args()
   
    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    dset_dir = args.savedset
    output_file = args.output
    if args.seg.lower() == "true":
        seg = True 
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.savemodel
    gpu = torch.cuda.is_available()
    # gpu = False
    ## disable cudnn to avoid memory leak
    # torch.backends.cudnn.enabled = True
    print "Seed num:",seed_num
    print "CuDNN:", torch.backends.cudnn.enabled
    # gpu = False
    print "GPU available:", gpu
    print "Status:", status
    print "Seg: ", seg
    print "Train file:", train_file
    print "Dev file:", dev_file
    print "Test file:", test_file
    print "Raw file:", raw_file
    if status == 'train':
        print "Model saved to:", save_model_dir
    sys.stdout.flush()
    
    if status == 'train':
        emb = args.wordemb.lower()
        print "Word Embedding:", emb
        if emb == "senna":
            emb_file = "data/SENNA.emb"
        elif emb == "glove":
            emb_file = "data/glove.6B.100d.txt"
        elif emb == "ctb":
            emb_file = "data/ctb.50d.vec"
        elif emb == "richchar":
            emb_file = None
            # emb_file = "../data/gigaword_chn.all.a2b.uni.ite50.vec"
            emb_file = "../data/joint4.all.b10c1.2h.iter17.mchar" 
        else:
            emb_file = None
        char_emb_file = args.charemb.lower()
        print "Char Embedding:", char_emb_file
        if char_emb_file == "rich":
            char_emb_file = "../data/joint4.all.b10c1.2h.iter17.mchar"
        elif char_emb_file == "normal":
            char_emb_file = "../data/gigaword_chn.all.a2b.uni.ite50.vec"

        data = Data()
        data.number_normalized = True
        data_initialization(data, train_file, dev_file, test_file)
        data.HP_gpu = gpu
        data.HP_use_char = False
        data.HP_batch_size = 1
        data.HP_lr = 0.015
        data.char_features = "CNN"
        data.generate_instance(train_file,'train')
        #data.generate_instance(dev_file,'dev')
        data.generate_instance(test_file,'test')
        if emb_file:
            print "load word emb file... norm:", data.norm_word_emb
            data.build_word_pretrain_emb(emb_file)
        if char_emb_file != "none":
            print "load char emb file... norm:", data.norm_char_emb
            data.build_char_pretrain_emb(char_emb_file)

        #model=pretrain(data, save_model_dir, seg)
        train(data, save_model_dir, seg)
    elif status == 'test':      
        data = load_data_setting(dset_dir)
        data.generate_instance(dev_file,'dev')
        load_model_decode(model_dir, data , 'dev', gpu, seg)
        data.generate_instance(test_file,'test')
        load_model_decode(model_dir, data, 'test', gpu, seg)
    elif status == 'decode':       
        data = load_data_setting(dset_dir)
        data.generate_instance(raw_file,'raw')
        decode_results = load_model_decode(model_dir, data, 'raw', gpu, seg)
        data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print "Invalid argument! Please use valid arguments! (train/test/decode)"





