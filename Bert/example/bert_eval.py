#coding:utf-8
import os
import sys
import csv
import codecs
import numpy as np

sys.path.append('../')

import tensorflow as tf
from preprocess import tokenization
from preprocess import bert_data_utils

#os.environ["CUDA_VISIBLE_DEVICES"] = "" # not use GPU


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("do_lower_case", True, "Whether to lower case the input text")
flags.DEFINE_string("vocab_file", "../chinese_L-12_H-768_A-12"+'/vocab.txt', "vocab file")
flags.DEFINE_string("label_map_file", "./bert_checkpoint"+'/label_map', "label map file")
flags.DEFINE_string("model_dir", "./bert_checkpoint/checkpoints", "vocab file")
tf.flags.DEFINE_string("test_data_file", "../data/test.txt", "Test data source.")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("top", 5, "predict top cid")
tf.flags.DEFINE_integer("max_sequence_length", 20, "max sequnce length")
tf.flags.DEFINE_float("threshold", 0.5, "threshold for predict")
tf.flags.DEFINE_string("cid3_file", '../data/cid3_name.txt', "cid3 name ")
tf.flags.DEFINE_string("save_file", '../data/predict.txt', "predict file ")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def get_cid_name(cid3_file):
    label_name = {}
    with codecs.open(cid3_file, encoding='utf8') as f:
        for line in f:
            line_list = line.strip().split('\t')
            label_name[line_list[1]] = line_list[2]
    return label_name

def eval():
    f = open(FLAGS.save_file, 'w+')
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    label_map, idx2label = bert_data_utils.read_label_map_file(FLAGS.label_map_file)
    label_name = get_cid_name(FLAGS.cid3_file)
    batch_datas = bert_data_utils.get_data_yield(FLAGS.test_data_file, 
                                                                                label_map,
                                                                                FLAGS.max_sequence_length,
                                                                                tokenizer,
                                                                                FLAGS.batch_size)

    print('\nEvaluating...\n')

    #Evaluation
    # checkpoint_file = tf.train.latest_checkpoint(FLAGS.model_dir)
    graph = tf.Graph()
    with graph.as_default():
        #restore for tensorflow pb style
        # restore_graph_def = tf.GraphDef()
        # restore_graph_def.ParseFromString(open(FLAGS.model_dir+'/frozen_model.pb', 'rb').read())
        # tf.import_graph_def(restore_graph_def, name='')

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        #restore for tf checkpoint style
        cp_file = tf.train.latest_checkpoint(FLAGS.model_dir)
        # saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('{}.meta'.format(cp_file))
        saver.restore(sess,cp_file)
        
        with sess.as_default():
            #tensors we feed
            input_ids = graph.get_operation_by_name('input_ids').outputs[0]
            input_mask = graph.get_operation_by_name('input_mask').outputs[0]
            token_type_ids = graph.get_operation_by_name('segment_ids').outputs[0]
            is_training = graph.get_operation_by_name('is_training').outputs[0]
            
            #tensors we want to evaluate
            # precision =  graph.get_operation_by_name('accuracy/precision').outputs[0]
            # recall = graph.get_operation_by_name('accuracy/recall').outputs[0]
            # f1 = graph.get_operation_by_name('accuracy/f1').outputs[0]
            predictions = graph.get_operation_by_name('loss/predictions').outputs[0]


            #collect the predictions here
            for batch in batch_datas:
                feed_input_ids, feed_input_mask, feed_segment_ids, querys = batch

                feed_dict = {input_ids: feed_input_ids,
                             input_mask: feed_input_mask,
                             token_type_ids: feed_segment_ids,
                             is_training: False,}

                batch_predictions = sess.run(predictions,feed_dict)
                for  prediction, query in zip(batch_predictions, querys):
                    predictions_sorted = sorted(prediction, reverse=True)
                    index_sorted = np.argsort(-prediction)
                    t =0
                    label_list = []
                    #label_scores = []
                    label_names = []
                    for index, predict in zip(index_sorted, predictions_sorted):
                        if predict >=FLAGS.threshold:
                            label = idx2label[index]
                            label_list.append(label+':'+str(predict))
                            #label_scores.append(str(predict))
                            label_names.append(label_name[label])
                    if len(label_list) == 0:
                        label_list.append('0:0')
                        #label_scores.append('0')
                        label_names.append(u'填充类')

                    f.write(query+'\t'+','.join(label_list)+'\t'+','.join(label_names)+'\n')


if __name__ == '__main__':
    eval()
