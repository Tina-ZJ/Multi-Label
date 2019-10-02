# Multi-Label
Multi label for text classification

#######  BiGRU+attention #######

Data format
query \t segwords \t label1,label2

Train:

1. cd BiGUU+attention

2. bash train.sh

Test:

1. python predit.py


#######  Bert #######

Data format
query \t label1,label2,label3

Prepare:

1. Download the google base model for chinese: https://github.com/google-research/bert

2. put the chinese_L-12_H-768_A-12 in Bert directory

Train:

1. cd Bert/example

2. bash train.sh

Test:

1.bash predict.sh
