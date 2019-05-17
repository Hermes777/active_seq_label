# -*- coding: utf-8 -*-
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
from utils.data import Data
sys.path.append("..")
sys.path.append(".")
from model.reinforcement_model import SeqModel

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
parser.add_argument('--seq_choosing',default="False") 
parser.add_argument('--randseed', default=100) 
parser.add_argument('--pre_iter_crf', default=5) 
parser.add_argument('--pre_iter', default=30) 
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


# def predict_check(pred_variable, gold_variable, mask_variable):
#     """
#         input:
#             pred_variable (batch_size, sent_len): pred tag result, in numpy format
#             gold_variable (batch_size, sent_len): gold result variable
#             mask_variable (batch_size, sent_len): mask variable
#     """
#     pred = pred_variable.cpu().data.numpy()
#     gold = gold_variable.cpu().data.numpy()
#     mask = mask_variable.cpu().data.numpy()
#     overlaped = (pred == gold)
#     right_token = np.sum(overlaped * mask)
#     total_token = mask.sum()
#     # print("right: %s, total: %s"%(right_token, total_token))
#     return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):

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
    model.examiner.eval()
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    if qleft==None:
        qleft=0
        qright=int(total_batch/10)
    # print("name",name,qright)
    if name=="test":
        print "start test"
    word_dict={}
    tot_dict={}
    for batch_id in range(qleft,qright):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end >train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        #print("batch_word",batch_word)
        tag_seq = model.test(batch_word)#porblem: why it contains zero
        #print "tag:",tag_seq
        #print "batch_label:",batch_label
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        batch_label,_tag_seq,_tag_prob,tag_mask,score,indices,scores_ref=model.pos_selection(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
        #print("gold",gold_label)
        #print("pred",pred_label)
        pred_results += pred_label
        gold_results += gold_label
        if name=="test":
            for i in range(batch_wordlen[0]):
                #w=data.word_alphabet.get_instance(batch_word[0].data[i]).encode('ascii','ignore')
                w=gold_label[0][i]
                #print "word", ' '.join([data.word_alphabet.get_instance(batch_word[0].data[i]) for i in range(batch_wordlen[0])])
                #print "pred", pred_label
                #print "gold", gold_label
                if tag_mask.data[0][i]==0:
                    #if w=="00:00.0":
                    # print "mask", tag_mask
                    if (w not in word_dict):
                        word_dict[w]=1
                    else:
                        word_dict[w]+=1
                if (w not in tot_dict):
                    tot_dict[w]=1
                else:
                    tot_dict[w]+=1
                #print(tag_mask.data[0]
    word_list=[w for w in word_dict]
    for w in word_list:
        #if tot_dict[w]>=30:
        word_dict[w]=(float(word_dict[w])/tot_dict[w],tot_dict[w])
        #else:
        #    word_dict.pop(w, None)
    # if name=="test":
    #     print("least")
    #     print(sorted(word_dict.items(), key=lambda item:item[1][0]) )
    #     print("most")
    #     print(sorted(word_dict.items(), key=lambda item:item[1][0]))
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    #print(gold_results)
    #print(qleft,qright)
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

# def sample(dist):
#     # dist is a tensor of shape (batch_size x max_time_steps x vocab_size)
#     choice = torch.multinomial(dist.view(-1, dist.size(2)), num_samples=1, replacement=True)
#     choice = choice.squeeze(1).view(*dist.size()[:2])
#     return choice

# def hamming_score(preds,labels):

#     tot = []
#     for (pred, label) in zip(preds.cpu(),labels.cpu()):
#         #print('pred',pred)
#         #print('label',label)
#         tot.append((1-(-label).ge(0).long()).mul((-torch.abs(pred-label)).ge(0).long()).sum().float().data/((1-(-label).ge(0).long()).sum().float().data))
#         #print('result',(1-(-label).ge(0).long()).mul((-torch.abs(pred-label)).ge(0).long()).sum().data)
#     return tot

def train(data, save_model_dir, seg=True):
    print "Training model..."
    data.show_data_summary()
    save_data_name = save_model_dir +".dset"
    save_data_setting(data, save_data_name)
    loss_function = nn.NLLLoss()
    model = SeqModel(data)
    #model=copy.deepcopy(premodel)
    optimizer = optim.SGD(model.examiner.parameters(), lr=data.HP_lr, momentum=data.HP_momentum)
    best_dev = -1
    data.HP_iteration = 5
    USE_CRF=True
    ## start training
    acc_list=[]
    p_list=[]
    r_list=[]
    f_list=[]
    map_list=[]
    #random.seed(2)
    print("total", )
    data.HP_lr=0.1
    epoch_start = time.time()
    temp_start = epoch_start

    instance_count = 0
    sample_id = 0
    sample_loss = 0
    total_loss = 0
    total_rl_loss = 0
    total_ml_loss = 0
    total_num = 0.0
    total_reward = 0.0
    right_token_reform = 0
    whole_token_reform = 0


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
    sum_click=0
    sum_p_at_5=0.0
    sum_p=1.0
    #if idx==0:
    #    selected_data=[batch_id for batch_id in range(0,total_batch//1000)]
    tag_mask=None
    batch_ids=[i for i in range(total_batch)]
    for batch_idx in range(0,total_batch):
        optimizer = lr_decay(optimizer, batch_idx, data.HP_lr_decay, data.HP_lr)
        batch_id=batch_ids[batch_idx]

        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end >train_num:
            end = train_num
        instance = data.train_Ids[start:end]
        if not instance:
            continue


        start_time = time.time()
        #selected_data.append(batch_id)

        if batch_id==args.pre_iter:

            for j in range(0,30):
                __tot=0.0
                for i in range(args.pre_iter_crf,args.pre_iter):
                    model.sample_train(0,i)
                    batch_id_temp=batch_ids[i]
                    start = batch_id_temp*batch_size
                    end = (batch_id_temp+1)*batch_size 
                    instance = data.train_Ids[start:end]

                    batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)

                    new_batch_label,tag_seq,tag_prob,tag_mask,score,indices,scores_ref=model.pos_selection(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)

                    #_pred_label, _gold_label = recover_label(Variable(tag_seq.cuda()), real_batch_label.cuda(),mask.cuda(), data.label_alphabet, batch_wordrecover)
                    _tag_mask=tag_mask

                    pos_mask,score  = model.reinforment_reward(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label,tag_seq,tag_prob, mask, mode="supervised_full")
                    __tot+=score.sum()

                    score.sum().backward()
                    optimizer.step()
                    model.examiner.zero_grad()

                print("score",__tot/(args.pre_iter-args.pre_iter_crf))
            model.train()
        if batch_id>=args.pre_iter:
            t=np.random.randint(0,len(model.X_train))
            if np.random.rand()>-1 or model.tag_mask_list[t].sum().data[0]<=topk:
                t=np.random.randint(len(model.X_train),total_batch)

                if args.seq_choosing==True:
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
                        new_batch_label,tag_seq,tag_mask,score,indices,scores_ref=model.pos_selection(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
                        if tmin==-1 or (scores_ref.cpu().data[0]>=tmin):
                            tmin=scores_ref.cpu().data[0]
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
                    

                batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
                new_batch_label,tag_seq,tag_prob,tag_mask,score,indices,scores_ref=model.pos_selection(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
                model.add_instance(batch_word,new_batch_label,tag_mask,instance,scores_ref.data[0])

            else:
                if args.seq_choosing==True:
                    tmin=model.scores_refs[t]
                    for i in range(len(model.X_train)):
                        if model.scores_refs[i]<=tmin:
                            tmin=model.scores_refs[i]
                            t=i

                instance = model.instances[t]
                batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
                new_batch_label,tag_seq,tag_prob,tag_mask,score,indices,scores_ref=model.pos_selection(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask, t=t)
                model.readd_instance(new_batch_label,mask,tag_mask,t,scores_ref.data[0])


                print("score",score)
                sum_p+=1.0

                end_time = time.time()
                if click+5>=10:
                    print("time",end_time-start_time)
        else:
            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
            model.add_instance(batch_word,batch_label,tag_mask,instance,-100000.0)
        

        if batch_id<args.pre_iter:
            if batch_id==args.pre_iter-1:
                model.train()
                print(batch_ids)
                speed, acc, p, r, f, _ = evaluate(data, model, "test")
                # print(len(model.Y_train))
                print("test accuracy",acc)
                # print("Check",f)
                acc_list.append(acc)
                p_list.append(p)
                r_list.append(r)
                f_list.append(sum_click)
                sum_p_at_5=0.0
                sum_p=1.0
            continue

        click+=model.topk
        sum_click+=model.topk
        #click+=batch_wordlen[0]
        #sum_click+=batch_wordlen[0]
        if click>=args.out_bucket:
            model.train()
            speed, acc, p, r, f, _ = evaluate(data, model, "test")
            print("Step:",len(model.Y_train))
            print("test accuracy",acc)
            acc_list.append(acc)
            p_list.append(p)
            r_list.append(r)
            f_list.append(sum_click)
            sum_p_at_5=0.0
            sum_p=1.0

            click-=args.out_bucket
        instance_count += 1

        pos_mask,selection_score = model.reinforment_reward(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label,tag_seq,tag_prob, mask,mode="reinforcement")
        if USE_CRF==True:
            start_time = time.time()
            t=np.random.randint(1,10)
            #print("size",total_batch)
            speed, acc, p, r, f, _ = evaluate(data, model, "dev")
            end_time = time.time()
            if total_num!=0:
                ave_scores=total_reward/total_num
            else:
                ave_scores=0.0
            total_reward+=acc
            total_num+=1

            # print(batch_label)
            sample_scores = torch.from_numpy(np.asarray([acc])).float()
            ave_scores= torch.from_numpy(np.asarray([ave_scores])).float()
            reward_diff = Variable(sample_scores-ave_scores, requires_grad=False)         
            reward_diff = reward_diff.cuda()
        rl_loss = -selection_score # B


        # print("reward",reward_diff)
        # print("rl_loss",rl_loss)
        rl_loss = torch.mul(rl_loss, reward_diff.expand_as(rl_loss))#b_size


        rl_loss.backward()
        optimizer.step()
        model.examiner.zero_grad()
        if len(p_list)>=100:
            break
    gc.collect() 
    file_dump=open("result_list.pkl","w")
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
        data.pre_iter_crf= args.pre_iter_crf
        data.pre_iter= args.pre_iter
        data.iter= args.pre_iter

        data.generate_instance(train_file,'train')
        data.generate_instance(dev_file,'dev')
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





