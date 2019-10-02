# -*-coding:utf8 -*-
import sys
import codecs

def term(data_file, label_file, label_name):
    f = codecs.open(data_file, encoding='utf8')
    f2 = codecs.open(label_file, 'w', encoding='utf8')
    f3 = codecs.open(label_name, 'w', encoding='utf8')
    labels = list()
    for line in f:
        terms = line.strip().split('\t')
        if len(terms)!=2:
            continue
        label = terms[1].split(',')
        for la in label:
            if la not in labels :
                labels.append(la)
    for i, la in enumerate(labels):
        f3.write(str(i+1) +'\t'+str(la)+'\t'+str(la)+'\n')
        f2.write(str(la)+'\n')

if __name__=='__main__':
    data_file = sys.argv[1]
    label_file = sys.argv[2]
    label_name = sys.argv[3]
    term(data_file, label_file, label_name) 
        
