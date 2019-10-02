#-*-coding=utf-8-*-


import codecs
import numpy as np



def get_cid_index(filename):
    cid_index = {}
    with open(filename) as f:
        for line in f:
            line = line.replace('\r\n', '\n')
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                cid_index[int(tokens[1])] = int(tokens[0])
    return cid_index

def get_index_cid(filename):
    index_cid = {}
    with open(filename) as f:
        for line in f:
            line = line.replace('\r\n', '\n')
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                index_cid[int(tokens[0])] = int(tokens[1])
    return index_cid


def get_term_index(filename):
    index_term = {}
    term_index = {}
    with open(filename) as f:
        for line in f:
            line = line.replace('\r\n', '\n')
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                if len(tokens)!=2:
                    print(line)
                index_term[int(tokens[1])] = tokens[0]
                term_index[tokens[0]] = int(tokens[1])
    return index_term, term_index

def get_one_hot_label(labels, class_num):
    one_hot_labels = []
    for label in labels:
        one_hot_label = [0.0] * class_num
        for lab in label:
            if int(lab) != 0:
                one_hot_label[int(lab)-1] = 1.0
        one_hot_labels.append(one_hot_label)
    return one_hot_labels

def load_term_embedding(embedding_file,term_index):
    emb = {}
    with open(embedding_file) as f:
        m, n = f.readline().split()
        for line in f:
            line = line.replace('\r\n', '\n')
            line = line.strip('\n')
            if line and len(line.split()) > 1:
                term = line.split()[0]
                vec = [float(v) for v in line.split()[1:]]
                if term in term_index:
                    emb[term_index[term]] = vec
    return emb

def get_term_embedding(embedding_file,term_index):
    emb = load_term_embedding(embedding_file, term_index)
    vocab_size = len(term_index)
    embedding = []
    dim = 0
    for k in emb:
        dim = len(emb[k])
        if dim != 0:
            break
    for i in range(vocab_size):
        if i in emb:
            embedding.append(emb[i])
        else:
            embedding.append(np.zeros(dim))
    return np.array(embedding)
