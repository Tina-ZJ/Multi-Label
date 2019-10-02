# -*- coding: utf8 -*-

startTime=`date +"%Y-%m-%d %H:%M:%S"`
start_seconds=$(date +%s)

python bert_eval.py

endTime=`date +"%Y-%m-%d %H:%M:%S"`
end_seconds=$(date +%s)
useSeconds=$[$end_seconds - $start_seconds]
useHours=$[$useSeconds / 3600]

echo " the script running time: $startTime --> $endTime : $useSeconds seconds "
