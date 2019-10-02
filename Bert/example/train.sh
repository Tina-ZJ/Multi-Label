#!/bin/bash
python ../preprocess/term.py ../data/train.tsv ../data/labels.tsv ../data/cid3_name.txt
python bert_train.py --task_name=test --train_batch_size=256 --type_feature=mean --learning_rate=0.0001 --eval_batch_size=256 --output_dir=./bert_checkpoint/ --data_dir=../data --init_checkpoint=../chinese_L-12_H-768_A-12/bert_model.ckpt --bert_config_file=../chinese_L-12_H-768_A-12/bert_config.json --vocab_file=../chinese_L-12_H-768_A-12/vocab.txt --max_seq_length=20  --do_train=true --num_train_epochs=50
