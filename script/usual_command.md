#后台运行
nohup bash pengzai_train.sh > train_log.txt 2>&1 &

# kill all YOUR_TRAINING_SCRIPT.py processing
kill $(ps aux | grep YOUR_TRAINING_SCRIPT.py | grep -v grep | awk '{print $2}')

kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')