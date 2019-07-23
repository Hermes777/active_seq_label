# -*- coding: utf-8 -*-
import time
import sys
import argparse
import numpy
import random
import copy
import torch
import gc
import pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from utils.metric import get_ner_fmeasure
from utils.data import Data
sys.path.append("..")
sys.path.append(".")
from model.reinforcement_model import SeqModel
import scipy
import scipy.stats

# parser definition
parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CRF')
parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train') # test/decode TBD

#  directory parser
parser.add_argument('--savemodel', default="data/saved_model.lstmcrf.")
parser.add_argument('--savedset', help='Dir of saved data setting', default="data/save.dset")

#  wordemb parser
parser.add_argument('--wordemb',  help='Embedding for words', default='glove')
parser.add_argument('--charemb',  help='Embedding for chars', default='None')

# dataset parser
parser.add_argument('--train', default="data/train.bmes") 
parser.add_argument('--dev', default="data/dev.bmes" )  
parser.add_argument('--test', default="data/test.bmes") 
parser.add_argument('--extendalphabet', default="True") 
parser.add_argument('--raw') 
parser.add_argument('--loadmodel')
parser.add_argument('--output') 

# training options parser
parser.add_argument('--metric', choices=["f1","accuracy"],default="accuracy") 
parser.add_argument('--pretraining', choices=['supervised', 'reinforcement'],default="reinforcement") 
parser.add_argument('--training', choices=['supervised', 'reinforcement'],default="reinforcement") 
parser.add_argument('--replay',default="False") 
parser.add_argument('--seq_choosing', choices=['topk', 'entropy','None'],default="None") 
parser.add_argument('--randseed', default=102) 
parser.add_argument('--pre_iter_crf', default=5) 
parser.add_argument('--pre_iter', default=15) 
parser.add_argument('--iter', default=100) 
parser.add_argument('--topk', default=5) 
parser.add_argument('--out_bucket', default=10) 

#parser.add_argument('--tagger', choices=['crf', 'lstmcrf']) TBD

args = parser.parse_args()

seed_num = args.randseed
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.fix_alphabet()

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input: pred and ground truth label sequences as tensor
        output: pred and ground truth label sequences as its original string
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
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)


def load_data_setting(save_file):
    with open(save_file, 'r') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate(data, model, name, qleft=None,qright=None,batch_size=1):
    """
        input: 
            name: our current dataset for evaluation
            qleft,qright: the start and end point of the validation data set. 
                          When the validation data set is huge, we can use these parameters to sample the dataset
        output:
            speed:
            acc: accuracy
            p:precision
            r:recall
            f:f1 score
            pred_results：the predict results as a list of string
            p,r,f are useful when you switch to NER dataset
    """
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    ## set model in eval mode
    model.examiner.train()
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    if qleft==None:
        qleft=0
        qright=int(total_batch/10)
    if name=="test":
        print("start test")

    for batch_id in range(qleft,qright):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end >train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)

        tag_seq = model.test(batch_word)

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        batch_label,_tag_seq,_tag_prob,tag_mask,score,indices,scores_ref=model.pos_selection(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)

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
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        #print(seqlen)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen.numpy())
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len.numpy()-len(chars[idx])) for idx in range(len(chars))]

    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
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
    print("Training model...")
    data.show_data_summary()
    save_data_name = save_model_dir +".dset"
    save_data_setting(data, save_data_name)
    loss_function = nn.NLLLoss()
    model = SeqModel(data)

    USE_CRF=True

    acc_list=[]
    p_list=[]
    r_list=[]
    f_list=[]
    map_list=[]

    data.HP_lr=0.0001
    epoch_start = time.time()
    temp_start = epoch_start


    total_num = 0.0
    total_reward = 0.0
    optimizer = optim.SGD(model.examiner.parameters(), lr=data.HP_lr, momentum=data.HP_momentum)


    model.examiner.train()
    model.examiner.zero_grad()
    model.topk=args.topk
    model.examiner.topk=args.topk
    batch_size = data.HP_batch_size
    batch_id = 0
    train_num = len(data.train_Ids)
    total_batch = train_num//batch_size+1
    gamma=0
    cnt=0
    click=0
    _max=0.0
    sum_click=0


    tag_mask=None
    batch_ids=[i for i in range(total_batch)]
    for batch_idx in range(0,total_batch):
        model.examiner.train()
        batch_id=batch_ids[batch_idx]

        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end >train_num:
            end = train_num
        instance = data.train_Ids[start:end]
        if not instance:
            continue


        start_time = time.time()

        
        if batch_id==args.pre_iter-1:
            model.train()
            #model.examiner.init_crf(model.crf)
            print("start")
            print("parameters",len(model.crf.state_features_))
            for j in range(0,30):
                __tot=0.0
                batch_ids[:args.pre_iter]=numpy.random.permutation(batch_ids[:args.pre_iter])
                for i in range(args.pre_iter_crf,args.pre_iter):
                    model.sample_train(0,i)
                    batch_id_temp=batch_ids[i]
                    start = batch_id_temp*batch_size
                    end = (batch_id_temp+1)*batch_size 
                    instance = data.train_Ids[start:end]

                    batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)

                    new_batch_label,tag_seq,tag_prob,tag_mask,score,indices,scores_ref=model.pos_selection(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)

                    #_pred_label, _gold_label = recover_label(Variable(tag_seq.cuda()), real_batch_label.cuda(),mask.cuda(), data.label_alphabet, batch_wordrecover)


                    pos_mask,score = model.reinforcement_reward(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label,tag_seq,tag_prob, mask, mode="supervised_full")
                    
                    # print("word",[data.word_alphabet.get_instance(x) for x in batch_word[0].data])
                    # print("string",batch_label)
                    # print("new_batch_label",new_batch_label)
                    # print(score)

                    # #print("crf strcuture",model.crf.state_features_)
                    # print("new",tag_seq)

                    __tot+=score.sum()
                    score.sum().backward()
                    optimizer.step()
                    model.examiner.zero_grad()

                print("score",__tot/(args.pre_iter-args.pre_iter_crf))
            model.train()
            optimizer = optim.SGD(model.examiner.parameters(), lr=data.HP_lr, momentum=data.HP_momentum)
        tot_pos=0

        if batch_idx>=args.pre_iter:
            t=np.random.randint(len(model.X_train),total_batch)
            ###
            #
            # Selecting the sequence
            #
            ###
            if args.seq_choosing=="topk":
                tmin=-1
                for i in range(len(model.X_train),total_batch):
                    batch_id=batch_ids[i]
                    start = batch_id*batch_size
                    end = (batch_id+1)*batch_size 
                    if end >train_num:
                        end = train_num
                    instance = data.train_Ids[start:end]
                    if len(instance)==0:
                        continue
                    batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
                    new_batch_label,tag_seq,tag_prob,tag_mask,score,indices,scores_ref,correct=model.pos_selection(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
                    if tmin==-1 or (scores_ref.cpu().data[0]>=tmin):
                        tmin=scores_ref.cpu().data[0]
                        t=i
                temp=batch_ids[batch_idx]
                batch_ids[batch_idx]=batch_ids[t]
                batch_ids[t]=temp
            elif args.seq_choosing=="entropy":
                tmin=-1

                for i in range(len(model.X_train),total_batch):
                    batch_id=batch_ids[i]
                    start = batch_id*batch_size
                    end = (batch_id+1)*batch_size 
                    if end >train_num:
                        end = train_num
                    instance = data.train_Ids[start:end]
                    if len(instance)==0:
                        continue
                    batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
                    entropy=model.compute_entropy([data.word_alphabet.get_instance(x) for x in batch_word[0].data])
                    #new_batch_label,tag_seq,tag_prob,tag_mask,score,indices,scores_ref=model.pos_selection(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
                    if tmin==-1 or entropy>=tmin:
                        tmin=entropy
                        t=i

                temp=batch_ids[batch_idx]
                batch_ids[batch_idx]=batch_ids[t]
                batch_ids[t]=temp

            batch_id=batch_ids[batch_idx]
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size 
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
                    
            ###
            #
            # Selecting the position
            #
            ###
            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
            new_batch_label,tag_seq,tag_prob,tag_mask,score,indices,scores_ref=model.pos_selection(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
            
            ###
            #
            # add the generated sequenc to the dataset
            #
            ###
            print("word",[data.word_alphabet.get_instance(x) for x in batch_word[0].data])
            print("string",batch_label)
            print("new_batch_label",new_batch_label)
            print("new",tag_seq)
            model.add_instance(batch_word,new_batch_label,tag_mask)


        else:
            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
            model.add_instance(batch_word,batch_label,tag_mask)
        

        if batch_idx<args.pre_iter:
            if batch_id==args.pre_iter-1:
                model.train()
                print(batch_ids)
                speed, acc, p, r, f, _ = evaluate(data, model, "test")
                # print(len(model.Y_train))
                print("test accuracy",acc)
                print("parameters",model.crf.state_features_)

                # print("Check",f)
                acc_list.append(acc)
                p_list.append(p)
                r_list.append(r)
                f_list.append(sum_click)

            continue

        click+=model.topk
        sum_click+=model.topk

        if click>=args.out_bucket:
            model.train()
            speed, acc, p, r, f, _ = evaluate(data, model, "test")
            print("Step:",len(model.Y_train))
            print("click",sum_click)
            print("test accuracy",acc)
            if acc>_max:
                _max=acc
            print("parameters",len(model.crf.state_features_))

            acc_list.append(acc)
            p_list.append(p)
            r_list.append(r)
            f_list.append(sum_click)

            click-=args.out_bucket

        ###
        #
        # Evaluate the model, obtaining the reward
        #
        ###
        start_time = time.time()
        t=np.random.randint(1,10)
        pos_mask,selection_score= model.reinforcement_reward(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label,tag_seq,tag_prob, mask,mode="reinforcement")

        model.train()
        speed, acc, p, r, f, _ = evaluate(data, model, "dev")
        end_time = time.time()
        if total_num!=0:
            ave_scores=total_reward/total_num
        else:
            ave_scores=0.0
        total_reward+=acc
        total_num+=1

        sample_scores = torch.from_numpy(np.asarray([acc])).float()
        ave_scores= torch.from_numpy(np.asarray([ave_scores])).float()
        reward_diff = Variable(sample_scores-ave_scores, requires_grad=False)
        if data.HP_gpu:
            reward_diff = reward_diff.cuda()

        #the selection score now is what we have in evaluation, we need to calculate it again
        model.examiner.train()

        ###
        #
        # Evaluate the model, obtaining the reward, backward
        #
        ###
        rl_loss = -selection_score 

        rl_loss = torch.mul(rl_loss, reward_diff[0])


        rl_loss.backward()
        # if batch_id>=args.pre_iter-1:
        #     optimizer = lr_decay(optimizer, 1, data.HP_lr_decay, data.HP_lr)
        optimizer.step()
        model.examiner.zero_grad()
        if len(p_list)>=100:
            break
    gc.collect() 
    print("max test accuracy",_max)
    file_dump=open("result_list.pkl","wb")
    pickle.dump([acc_list,p_list,r_list,f_list,map_list],file_dump)
    file_dump.close()

    #print("Best Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(best_[0], best_[1], best_[2], best_[3], best_[4], best_[5]))



def load_model_decode(model_dir, data, name, gpu, seg=True):
    data.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
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
   
    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    dset_dir = args.savedset
    output_file = args.output

    if args.metric == "f1":
        seg = True 
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.savemodel
    gpu = torch.cuda.is_available()
    # gpu = False
    ## disable cudnn to avoid memory leak
    # torch.backends.cudnn.enabled = True
    print("Seed num:",seed_num)
    print("CuDNN:", torch.backends.cudnn.enabled)
    # gpu = False
    print("GPU available:", gpu)
    print("Status:", status)
    print("Seg: ", seg)
    print("Train file:", train_file)
    print("Dev file:", dev_file)
    print("Test file:", test_file)
    print("Raw file:", raw_file)
    if status == 'train':
        print("Model saved to:", save_model_dir)
    sys.stdout.flush()
    
    if status == 'train':
        emb = args.wordemb.lower()
        print("Word Embedding:", emb)
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
        print("Char Embedding:", char_emb_file)
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
        data.pre_iter_crf= args.pre_iter_crf
        data.pre_iter= args.pre_iter
        data.iter= args.pre_iter

        data.generate_instance(train_file,'train')
        data.generate_instance(dev_file,'dev')
        data.generate_instance(test_file,'test')
        if emb_file:
            print("load word emb file... norm:", data.norm_word_emb)
            data.build_word_pretrain_emb(emb_file)
        if char_emb_file != "none":
            print("load char emb file... norm:", data.norm_char_emb)
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
        print("Invalid argument! Please use valid arguments! (train/test/decode)")





