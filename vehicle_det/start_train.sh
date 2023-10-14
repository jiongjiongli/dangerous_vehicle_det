
cd /project/train/src_repo/ultralytics
export PYTHONPATH=$PYTHONPATH:/project/train/src_repo/ultralytics

echo 'Reset env...'
mkdir -p /project/train/models

rm -rf /project/train/tensorboard/*
mkdir -p /project/train/tensorboard

rm -rf /project/train/log/*
mkdir -p /project/train/log

rm -rf /project/train/result-graphs/*
mkdir -p /project/train/result-graphs

echo 'Start data_config...'
python vehicle_det/data_config.py

echo 'Start cvmart_train...'
python vehicle_det/yolo_train.py

echo 'Completed!'
