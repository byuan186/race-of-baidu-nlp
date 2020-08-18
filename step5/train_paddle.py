#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:13:42 2020

@author: guest
"""

import sys
import os
import copy
import pandas as pd
import numpy as np
import codecs
import json
import jieba
#from gensim.models import Word2Vec
import paddle.v2 as paddle
import matplotlib.pyplot as plt

# 参数定义
start_token = u"<s>"
end_token = u"<e>"
unk_token = u"<unk>"

start_id = 0
end_id = 1
unk_id = 2

# 词典路径
min_count = 5
vocab_path = "vocab_m%s.txt" %(min_count)

# problem，conversation 和 report的最大长度
# 在训练中，超过长度的样本会被剔除
# 在测试中，超过长度的样本会被截断
max_problem_len = 100
max_conversation_len = 800
max_report_len = 100

# 词向量和隐层节点维度
embedding_size=hidden_size=512

# 词向量文件路径
word2vec_file="gensim.sg.%d.txt" %(embedding_size)

# 预处理后文件存储路径
data_path = "data_set_m%s.json" %(min_count)

do_segmentation_flag = False

def seg_line(line):
    tokens = jieba.cut(line, cut_all=False)
    return " ".join(tokens)
    
def proc(line):
    tokens = line.split("|")
    result = []
    for t in tokens:
        result.append(seg_line(t))
    return " | ".join(result)
    
if do_segmentation_flag:
    # load original data
    train = pd.read_csv("train_set.csv")
    dev_1 = pd.read_csv("1st_round_dev_set.csv")
    test_1 = pd.read_csv("1st_round_test_set.csv")
    dev = pd.read_csv("final_round_dev_set.csv")
    test = pd.read_csv("final_round_test_set.csv")
    
    # segmentation
    for k in ['Brand', 'Collection', 'Problem', 'Conversation']:
        print k
        train[k] = train[k].apply(proc)
        dev[k] = dev[k].apply(proc)
        test[k] = test[k].apply(proc)

        dev_1[k]=dev_1[k].apply(proc)
        test_1[k]=test_1[k].apply(proc)
    
    num=len(train['Report'] )
    for i in range(num):

        if(isinstance(train['Report'][i],float)):
            train['Report'][i]=" "
        
        
    
    train['Report'] = train['Report'].apply(proc)
    dev['Report'] = dev['Report'].apply(proc)

    dev_1['Report'] = dev_1['Report'].apply(proc)
    
    # save segmented data
    train.to_csv("train_set.seg.csv", index=False, encoding='utf-8')
    dev.to_csv("final_round_dev_set.seg.csv", index=False, encoding='utf-8')
    test.to_csv("final_round_test_set.seg.csv", index=False, encoding='utf-8')


    dev_1.to_csv('1st_round_dev_set.seg.csv',index=False, encoding='utf-8')
    test_1.to_csv('1st_round_test_set.seg.csv',index=False, encoding='utf-8')

build_vocab_flag = False

def stat_dict(lines):
    word_dict = {}
    for line in lines:
        tokens = line.split(" ")
        for t in tokens:
            t = t.strip()
            if t:
                word_dict[t] = word_dict.get(t,0) + 1
    return word_dict

def filter_dict(word_dict, min_count=3):
    out_dict = copy.deepcopy(word_dict)
    for w,c in out_dict.items():
        if c < min_count:
            del out_dict[w]
    return out_dict

def build_vocab(lines, min_count=3):
    word_dict = stat_dict(lines)
    word_dict = filter_dict(word_dict, min_count)
    sorted_dict = sorted(word_dict.items(), key=lambda x:x[1], reverse=True)
    sorted_words = [w for w,c in sorted_dict]
    sorted_words = [start_token, end_token, unk_token] + sorted_words
    vocab = dict([(w,i) for i,w in enumerate(sorted_words)])
    reverse_vocab = dict([(i,w) for i,w in enumerate(sorted_words)])
    return vocab, reverse_vocab
    
def save_vocab(vocab, path):
    output = codecs.open(path, "w", "utf-8")
    for w,i in sorted(vocab.items(), key=lambda x:x[1]):
        output.write("%s %d\n" %(w,i))
    output.close()
    
def load_vocab(path):
    vocab = {}
    input = codecs.open(path, "r", "utf-8")
    lines = input.readlines()
    input.close()
    for l in lines:
        w, c = l.strip().split(" ")
        vocab[w] = int(c)
    return vocab
    
if build_vocab_flag:
    train = pd.read_csv("train_set.seg.csv", encoding='utf-8')
    
    lines = []
    for k in ['Problem', 'Conversation', 'Report']:
        lines.extend(list(train[k].values))
        
    vocab, reverse_vocab = build_vocab(lines, min_count)
    
    save_vocab(vocab, vocab_path)
    vocab = load_vocab(vocab_path)  

train_word2vec_flag=False

if train_word2vec_flag:
    train = pd.read_csv("train_set.seg.csv", encoding='utf-8')
    dev = pd.read_csv("final_round_dev_set.seg.csv", encoding='utf-8')
    test = pd.read_csv("final_round_test_set.seg.csv", encoding='utf-8')
    
    lines = []
    for k in ['Problem', 'Conversation', 'Report']:
        lines.extend(list(train[k].apply(lambda x:x.split(" ")).values))
        lines.extend(list(dev[k].apply(lambda x:x.split(" ")).values))

    for k in ['Problem', 'Conversation']:
        lines.extend(list(test[k].apply(lambda x:x.split(" ")).values))

    model = Word2Vec(lines, size=embedding_size, window=5, sg=1, min_count=5, workers=4)
    model.wv.save_word2vec_format(word2vec_file, binary=False)

transform_data_flag=False

def read_data(df, max_problem_len, max_conversation_len, max_report_len):
    problem_lens = df['Problem'].apply(lambda x: len(x.split(" ")))
    conversation_lens = df['Conversation'].apply(lambda x: len(x.split(" ")))
    report_lens = df['Report'].apply(lambda x: len(x.split(" ")))
    data = []
    for i in range(len(df)):
        if problem_lens[i] > max_problem_len or             conversation_lens[i] > max_conversation_len or             report_lens[i] > max_report_len:
            continue
        item = df.iloc[i]
        data.append([[start_token] + item['Problem'].split(" ") + [end_token], 
                [start_token] + item['Conversation'].split(" ") + [end_token], 
                [start_token] + item['Report'].split(" ") + [end_token]])
    return data
    
def read_test_data(df):
    data = []
    for i in range(len(df)):
        item = df.iloc[i]
        problem_vec = item['Problem'].split(" ")[0:max_problem_len]
        conversation_vec = item['Conversation'].split(" ")[0:max_conversation_len]
        data.append([[start_token] + problem_vec + [end_token], 
                [start_token] + conversation_vec + [end_token]])
    return data
    
def transform_data(data, vocab):
    # transform sent to ids
    out_data = []
    for d in data:
        tmp_d = []
        for sent in d:
            tmp_d.append([vocab.get(t, unk_id) for t in sent if t])
        out_data.append(tmp_d)
    return out_data
    
if transform_data_flag:
    train = pd.read_csv("train_set.seg.csv", encoding='utf-8')
    dev1 = pd.read_csv("1st_round_dev_set.seg.csv", encoding='utf-8')
    dev2 = pd.read_csv("final_round_dev_set.seg.csv", encoding='utf-8')
    vocab = load_vocab(vocab_path)
    
    new_dev = dev2[~dev2['QID'].isin(dev1['QID'].values)]
    new_dev = new_dev.reset_index(drop=True)

    train_set = read_data(train, max_problem_len, max_conversation_len, max_report_len)
    train_set.extend(
            read_data(new_dev, max_problem_len, max_conversation_len, max_report_len))
    dev_set = read_data(dev1, max_problem_len, max_conversation_len, max_report_len)

    print len(train_set)
    print len(dev_set)

    train_set_ids = transform_data(train_set, vocab)
    # write data
    output = open(data_path, "w")
    json.dump({"train": train_set_ids}, output)
    output.close()

with_gpu=True
print "with gpu: ", with_gpu

def save_model(trainer, save_path):
    with open(save_path, 'w') as f:
        trainer.save_parameter_to_tar(f)

def load_word2vec_file(word2vec_file):
    word2vec_dict = {}
    input = codecs.open(word2vec_file, "r", "utf-8")
    lines = input.readlines()
    input.close()
    word_num, dim = lines[0].split(" ")
    word_num = int(word_num)
    dim = int(dim)
    
    lines = lines[1:]
    for l in lines:
        l = l.strip()
        tokens = l.split(" ")
        if len(tokens) != dim + 1:
            continue
        w = tokens[0]
        v = np.array(map(lambda x:float(x), tokens[1:]))
        word2vec_dict[w] = v
    return word2vec_dict, dim


def seq_to_seq_net(vocab_size,
                   is_generating,
                   word_vector_dim=100,
                   hidden_size=100,
                   beam_size=3,
                   max_length=100,
                   dropout_rate=0.):

    # Network Architecture
    decoder_size = hidden_size  # dimension of hidden unit of GRU decoder
    encoder_size = hidden_size  # dimension of hidden unit of GRU encoder

    embedding_param = paddle.attr.ParamAttr(name='embedding')

    def embedding_layer(word_id):
        embed = paddle.layer.embedding(
            input=word_id, size=word_vector_dim, param_attr=embedding_param)
        return paddle.layer.dropout(input=embed, dropout_rate=dropout_rate)


    def BiGru_layer(embedding):
        forward = paddle.networks.simple_gru2(
            input=embedding, size=encoder_size, 
            gru_param_attr=paddle.attr.ParamAttr(name='gru_forward_encoder'),
            #gru_bias_attr=False,
            #mixed_bias_attr=False,
            mixed_param_attr=paddle.attr.ParamAttr(name='mixed_forward_encoder'))
            
        backward = paddle.networks.simple_gru2(
            input=embedding, size=encoder_size, reverse=True,
            gru_param_attr=paddle.attr.ParamAttr(name='gru_backward_encoder'),
            #gru_bias_attr=False,
            #mixed_bias_attr=False,
            mixed_param_attr=paddle.attr.ParamAttr(name='mixed_backward_encoder'))
        return forward, backward


    #### Encoder
    ##### Encode problem to a fixed size vector
    problem_word_id = paddle.layer.data(
        name='problem_word',
        type=paddle.data_type.integer_value_sequence(vocab_size))

    problem_embedding = embedding_layer(problem_word_id)

    problem_f, problem_b = BiGru_layer(problem_embedding)
    problem_f_last = paddle.layer.last_seq(input=problem_f)
    problem_b_first = paddle.layer.first_seq(input=problem_b)

    problem_vector = paddle.layer.concat(input=[problem_f_last, problem_b_first])
    problem_vector = paddle.layer.dropout(input=problem_vector, dropout_rate=dropout_rate)

    ##### Encode conversation
    conversation_word_id = paddle.layer.data(
        name='conversation_word',
        type=paddle.data_type.integer_value_sequence(vocab_size))

    conversation_embedding = embedding_layer(conversation_word_id)
    conversation_f, conversation_b = BiGru_layer(conversation_embedding)
    conversation_vector = paddle.layer.concat(input=[conversation_f, conversation_b])

    #### Decoder
    conversation_proj = paddle.layer.fc(
        act=paddle.activation.Linear(),
        size=decoder_size,
        bias_attr=False,
        input=conversation_vector)

    backward_first = paddle.layer.first_seq(input=conversation_b)

    decoder_boot = paddle.layer.fc(
        size=decoder_size,
        act=paddle.activation.Tanh(),
        bias_attr=False,
        input=backward_first)

    def gru_decoder_with_attention(
            enc_vec, enc_proj, problem_vec, current_word):

        decoder_mem = paddle.layer.memory(
            name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)

        context = paddle.networks.simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        decoder_inputs = paddle.layer.fc(
            act=paddle.activation.Linear(),
            size=decoder_size * 3,
            bias_attr=False,
            input=[context, problem_vec, current_word],
            layer_attr=paddle.attr.ExtraLayerAttribute(
                error_clipping_threshold=100.0))

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)
        
        out = paddle.layer.mixed(size=vocab_size,
            act=paddle.activation.Softmax(),
            bias_attr=False,
            ## 这里使用 embedding 矩阵参数 产生 report 的词
            input=paddle.layer.trans_full_matrix_projection(
                        input=gru_step,
                        param_attr=embedding_param))
        
        return out

    decoder_group_name = 'decoder_group'
    group_input1 = paddle.layer.StaticInput(input=conversation_vector)
    group_input2 = paddle.layer.StaticInput(input=conversation_proj)
    group_input3 = paddle.layer.StaticInput(input=problem_vector)
    group_inputs = [group_input1, group_input2, group_input3]

    if not is_generating:
        report_embedding = embedding_layer(paddle.layer.data(
            name='report_word',
            type=paddle.data_type.integer_value_sequence(vocab_size)))

        group_inputs.append(report_embedding)

        decoder = paddle.layer.recurrent_group(
            name=decoder_group_name,
            step=gru_decoder_with_attention,
            input=group_inputs)

        lbl = paddle.layer.data(
            name='report_next_word',
            type=paddle.data_type.integer_value_sequence(vocab_size))
        cost = paddle.layer.classification_cost(input=decoder, label=lbl)

        return cost
    else:
        report_embedding = paddle.layer.GeneratedInput(
            size=vocab_size,
            embedding_name='embedding',
            embedding_size=word_vector_dim)
        group_inputs.append(report_embedding)

        beam_gen = paddle.layer.beam_search(
            name=decoder_group_name,
            step=gru_decoder_with_attention,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=max_length)

        return beam_gen

def infer(inferer, reverse_vocab, data, batch_size=32, beam_size=3):
    def _infer_a_batch(inferer, reverse_vocab, batch_data, beam_size=3):
        beam_result = inferer.infer(
            input=batch_data,
            field=['prob', 'id'])

        gen_sen_idx = np.where(beam_result[1] == -1)[0]
        assert len(gen_sen_idx) == len(batch_data) * beam_size

        # -1 is the delimiter of generated sequences.
        # the first element of each generated sequence its length.
        batch_out = []
        start_pos, end_pos = 1, 0
        for i, sample in enumerate(batch_data):
            for j in xrange(beam_size):
                end_pos = gen_sen_idx[i * beam_size + j]
                #print("%.4f\t%s" % (beam_result[0][i][j], " ".join(
                #    reverse_vocab[w] for w in beam_result[1][start_pos:end_pos])))
                if j == 0:
                    batch_out.append("".join(reverse_vocab[w] for w in beam_result[1][start_pos:end_pos-1]))
                start_pos = end_pos + 2
        del beam_result
        beam_result = None
        return batch_out

    infer_results = []
    test_batch = []
    for idx, item in enumerate(data):
        test_batch.append(item)
        if len(test_batch) == batch_size:
            test_results = _infer_a_batch(inferer, reverse_vocab, test_batch, beam_size=beam_size)
            infer_results.extend(test_results)
            test_batch = []
            sys.stdout.write('.')
            sys.stdout.flush()

    if len(test_batch):
        test_results = _infer_a_batch(inferer, reverse_vocab, test_batch, beam_size=beam_size)
        infer_results.extend(test_results)
        test_batch = []

    return infer_results


def main(vocab, is_generating=False, model_path=None, w2v_path=None, 
        embedding_size=100, hidden_size=100, lr=5e-4, batch_size=32, n_epoch=2,
        out_model_prefix=None, data_name="dev", out_path="dev_result.csv",
        max_length=100, dropout_rate=0.):
    
    paddle.init(use_gpu=with_gpu, trainer_count=1)
    
    # load the dictionary
    reverse_vocab = dict([(i,w) for w,i in vocab.items()])
    
    # source and target dict dim.
    dict_size = len(vocab)
    vocab_size = vocab_size = dict_size
    
    # train the network
    if not is_generating:
        # define optimize method and trainer
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr,
            regularization=paddle.optimizer.L2Regularization(rate=8e-4))

        cost = seq_to_seq_net(vocab_size, is_generating, word_vector_dim=embedding_size,
            hidden_size=hidden_size, dropout_rate=dropout_rate)
        if model_path:
            parameters = paddle.parameters.Parameters.from_tar(
                open(model_path, "r"))
        else:
            parameters = paddle.parameters.create(cost)
            if w2v_path:
                word2vec_dict, dim = load_word2vec_file(w2v_path)
                assert dim == embedding_size

                embedding_matrix = np.random.normal(loc=0, scale=0.1, size=(len(vocab), dim))
                for word, i in vocab.items():
                    embedding_vector = word2vec_dict.get(word)
                    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
                
                parameters.set('embedding', embedding_matrix)
                del word2vec_dict
                word2vec_dict = None

        trainer = paddle.trainer.SGD(
            cost=cost, parameters=parameters, update_equation=optimizer)
        # define data reader
        qichedashi_reader = paddle.batch(
            paddle.reader.shuffle(
                train_reader, buf_size=100000),
            batch_size=batch_size)

        # define event_handler callback
        def event_handler(event):
            save_every = 0
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 30 == 0:
                    print("\nPass %d, Batch %d, Cost %f, %s" %
                          (event.pass_id, event.batch_id, event.cost,
                           event.metrics))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                
                if save_every > 0 and not event.batch_id % save_every:
                    last_save_path = '%s_params_pass_%05d_batch_%05d.tar' % (
                        out_model_prefix, event.pass_id, event.batch_id - save_every)
                    save_path = '%s_params_pass_%05d_batch_%05d.tar' % (
                        out_model_prefix, event.pass_id, event.batch_id)
                    save_model(trainer, save_path)
                    if event.batch_id > 0:
                        os.remove(last_save_path)
                

            if isinstance(event, paddle.event.EndPass):
                # save parameters
                save_path = '%s_params_pass_%05d.tar' % (out_model_prefix, event.pass_id)
                save_model(trainer, save_path)

        # start to train
        trainer.train(
            reader=qichedashi_reader, event_handler=event_handler, num_passes=n_epoch)

    else:
        # 输出三个样例看看
        # use the first 3 samples for generation
        gen_data = []
        real_data = []
        gen_num = 3
        for item in train_reader():
            gen_data.append([item[0],item[1]])
            real_data.append([item[0],item[1],item[2]])
            if len(gen_data) == gen_num:
                break

        beam_size = 3
        beam_gen = seq_to_seq_net(vocab_size, is_generating, word_vector_dim=embedding_size,
            hidden_size=hidden_size, beam_size=beam_size, max_length=max_length,
            dropout_rate=dropout_rate)

        # load the trained models
        parameters = paddle.parameters.Parameters.from_tar(
            open(model_path, "r"))

        # prob is the prediction probabilities, and id is the prediction word.
        beam_result = paddle.infer(
            output_layer=beam_gen,
            parameters=parameters,
            input=gen_data,
            field=['prob', 'id'])
            
        gen_sen_idx = np.where(beam_result[1] == -1)[0]
        assert len(gen_sen_idx) == len(gen_data) * beam_size

        # -1 is the delimiter of generated sequences.
        # the first element of each generated sequence its length.
        start_pos, end_pos = 1, 0
        for i, sample in enumerate(real_data):
            print(
                " ".join([reverse_vocab[w] for w in sample[0][1:-1]])
            )  # skip the start and ending mark when printing the source sentence
            print(
                " ".join([reverse_vocab[w] for w in sample[1][1:-1]])
            )
            print(
                " ".join([reverse_vocab[w] for w in sample[2]])
            )
            for j in xrange(beam_size):
                end_pos = gen_sen_idx[i * beam_size + j]
                print("%.4f\t%s" % (beam_result[0][i][j], " ".join(
                    reverse_vocab[w] for w in beam_result[1][start_pos:end_pos])))
                start_pos = end_pos + 2
            print("\n")

        # 在测试集/开发集 上输出
        # inference for data
        df = pd.read_csv("%s_set.seg.csv" %(data_name), encoding='utf-8')
        data = read_test_data(df)
        data = transform_data(data, vocab)
        
        inferer = paddle.inference.Inference(
            output_layer=beam_gen, parameters=parameters)
        infer_results = infer(inferer, reverse_vocab, data, batch_size=32, beam_size=3)
        #for r in infer_results:
        #    print(r)

        submit = df[['QID']]
        submit['Prediction'] = infer_results
        submit.to_csv(out_path, encoding="utf-8", index=False)
def reader_creator(data):
    def reader():
        for p_ids, c_ids, r_ids in data:
            yield p_ids, c_ids, r_ids[:-1], r_ids[1:]
    return reader
# load vocab
vocab = load_vocab(vocab_path)

# load data
input = open(data_path)
data = json.load(input)
input.close()

train_set_ids = data['train']
#dev_set_ids = data['dev']
#test_set_ids = data['test']
del data

train_reader = reader_creator(train_set_ids)

# 第一次训练
first_train_flag = False
if first_train_flag:
    # main(vocab, w2v_path=word2vec_file, embedding_size=embedding_size, hidden_size=hidden_size, 
    #     out_model_prefix="/home/aistudio/ans/d%d" %(hidden_size), n_epoch=7, batch_size=24)
    main(vocab, w2v_path=word2vec_file, embedding_size=embedding_size, hidden_size=hidden_size, 
        out_model_prefix="/home/aistudio/ans1/d%d" %(hidden_size), n_epoch=10, batch_size=32)    

# 持续训练
# continue_train_flag = False
# if continue_train_flag:
#     # main(vocab, model_path='ans/d512_p4_4_30_params_pass_00009.tar', out_model_prefix="/home/aistudio/ans/d512_p4_4_30",
#     #     embedding_size=embedding_size, hidden_size=hidden_size, 
#     #     n_epoch=10, batch_size=24, lr=1e-4)
#     main(vocab, model_path='ans1/d512_params_pass_00009.tar', out_model_prefix="/home/aistudio/ans1/d512_p4_4_30",
#         embedding_size=embedding_size, hidden_size=hidden_size, 
#         n_epoch=20, batch_size=32, lr=5e-5)

continue_train_flag = False
if continue_train_flag:
    # main(vocab, model_path='ans/d512_p4_4_30_params_pass_00009.tar', out_model_prefix="/home/aistudio/ans/d512_p4_4_30",
    #     embedding_size=embedding_size, hidden_size=hidden_size, 
    #     n_epoch=10, batch_size=24, lr=1e-4)
    # main(vocab, model_path='ans1/d512_p4_4_30_params_pass_00019.tar', out_model_prefix="/home/aistudio/ans2/d512_p4_4_30",
    #     embedding_size=embedding_size, hidden_size=hidden_size, 
    #     n_epoch=5, batch_size=32, lr=5e-5)
    main(vocab, model_path='ans/best/d512_p4_4_30_params_pass_00004.tar', out_model_prefix="/home/aistudio/ans/best/1/d512_p4_4_30",
        embedding_size=embedding_size, hidden_size=hidden_size, 
        n_epoch=5, batch_size=32, lr=5e-5)    

# predict
predict_flag = True
if predict_flag:
    data_name="final_round_test"
    # out_path="~/ans/t4_30_1.csv"
    # main(vocab, is_generating=True, model_path="ans/d512_p4_4_30_params_pass_00008.tar", embedding_size=embedding_size, hidden_size=hidden_size,
    #     data_name=data_name, out_path=out_path, max_length=100)
    out_path="~/ans/2/t4_30_00.csv"
    main(vocab, is_generating=True, model_path="ans/2/d512_p4_4_30_params_pass_00004 (4).tar", embedding_size=embedding_size, hidden_size=hidden_size,
        data_name=data_name, out_path=out_path, max_length=100)




