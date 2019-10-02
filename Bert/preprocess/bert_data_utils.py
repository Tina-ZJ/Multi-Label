# -*-coding:utf-8 -*-
import sys
import codecs

sys.path.append('../')

from preprocess import tokenization

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def read_code_file(code_file):
    #get real label
    label2code = {}
    with codecs.open(code_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip()
            line_info = line.split('\t')
            label = line_info[0].lower()
            code_value = line_info[1].lower()
            label2code[label] = code_value
    return label2code


def read_bert_labels_file(label_file):
    label_list = []
    with codecs.open(label_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip().lower()
            label_list.append(line)

    label_list = list(set(label_list))
    label2idx = {}
    idx2label = {}
    for (i, label) in enumerate(label_list):
        label2idx[label] = i
        idx2label[i] = label
        print(i, label)
    return label2idx, idx2label

def get_one_hot_label(label, class_num):
    one_hot_label = [0.0] * class_num
    for lab in label:
        one_hot_label[int(lab)] = 1.0
    return one_hot_label

def read_label_map_file(label_map_file):
    label_map = {}
    idx2label = {}
    with codecs.open(label_map_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip().lower()
            line_info = line.split('\t')
            idx = int(line_info[0])
            label = line_info[1]
            label_map[label] = idx
            idx2label[idx] = label
    return (label_map, idx2label)

def get_data_yield(data_file, label_map, max_seq_length, tokenizer, batch_size):
    B_input_ids, B_input_mask, B_segment_ids, querys = [], [], [], []
    count = 0
    for example in open(data_file, 'r'):
        terms = example.strip().split('\t')
        count+=1
        tokens_a = tokenizer.tokenize(terms[0])
        #if len(tokens_a) > max_seq_length -2:
            #tokens_a = tokens_a[0:(max_seq_length -2)]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] *len(input_ids)
        #while len(input_ids) < max_seq_length:
            #input_ids.append(0)
            #input_mask.append(0)
            #segment_ids.append(0)
        B_input_ids.append(input_ids)
        B_input_mask.append(input_mask)
        B_segment_ids.append(segment_ids)
        querys.append(terms[0])
        if count % batch_size ==0:
            yield(B_input_ids, B_input_mask, B_segment_ids, querys)
            B_input_ids, B_input_mask, B_segment_ids, querys = [], [], [], [] 



def get_data_from_file(file_name):
    text_trunk = []
    with codecs.open(file_name, 'r', 'utf8') as fr:
        for i, line in enumerate(fr):
            line = line.strip().lower()
            line_info = line.split('\t')
            text = line_info[0].strip()
            # label = line_info[1].strip()
            yield InputExample(guid=i, text_a=text)

def get_test(file_name):
    with codecs.open(file_name, 'r', 'utf8') as f:
        result = list()
        for line in f:
            result.append(line.strip())
    return result




def file_based_convert_examples_to_features(file_name, label_map, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    examples = get_data_from_file(file_name)
    test_data = get_test(file_name)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d" % (ex_index))

        feature = convert_single_example(ex_index, example, label_map,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return test_data, features
