# Multi-Label
Multi label for text classification

######################  BiGRU+attention #################

Data format
1. query \t segwords \t label1,label2
2. more details please see data/all_sample.txt samples

Train
1. cd BiGUU+attention
2. bash train.sh

Test
1. python predit.py


#######################  Bert ############################

Data format
1. query \t label1,label2,label3
2. more details please see data/train.tsv

Prepare
1. Download the google base model for chinese: https://github.com/google-research/bert
2. put the chinese_L-12_H-768_A-12 in Bert directory

Train
1. cd Bert/example
2. bash train.sh

Test
1. bash predict.sh
