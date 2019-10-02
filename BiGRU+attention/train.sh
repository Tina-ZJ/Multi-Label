#!/bin/bash

startTime=`date +"%Y-%m-%d %H:%M:%S"`
start_seconds=$(date +%s)

# data process
python preprocess/term.py data/all_sample.txt data/term_index.txt data/cid3_name.txt data/char_index.txt
if [ $? -ne 0 ]
then
    echo "preprocess data failed "
    exit -1
else
    echo "preprocess data Done "
fi

# split corpus to train and dev
python split_sample.py
if [ $? -ne 0 ]
then
    echo "split corpus to train and dev failed "
    exit -1
else
    echo "split corpus to train and dev Done "
fi
# transfer data to tfrecord format
python -u transfer_sample_tfrecord.py
if [ $? -ne 0 ]
then
    echo "transfer data to tfrecord failed"
    exit -1
else
    echo "transfer data to tfrecord Done "
fi

num_classes=`awk -F'\t' 'END{print $1}' data/cid3_name.txt`
train_sample_num=`cat data/train_sample.txt | wc -l`
dev_sample_num=`cat data/dev_sample.txt | wc -l`

# begain train model
python -u train.py --num_classes=${num_classes} --train_sample_num=${train_sample_num} --dev_sample_num=${dev_sample_num}
if [ $? -ne 0 ]
then
    echo " train HAN model failed"
    exit -1
else
    echo "train HAN model Done"
fi

endTime=`date +"%Y-%m-%d %H:%M:%S"`

# excute time
end_seconds=$(date +%s)
useSeconds=$[$end_seconds - $start_seconds]
useHours=$[$useSeconds / 3600]

echo " the script running time: $startTime ---> $endTime : $useHours hours "

