

cd /project/train/src_repo

# Download from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget -P /project/train/src_repo/ https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-36511-files/50c66f15-f262-4896-a72c-b66099b93421/yolov8n.pt

# Download from https://ultralytics.com/assets/Arial.ttf
wget -P /project/train/src_repo/ https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-36511-files/b37ad34b-5c66-4ede-b2f3-1dbbc7628c33/Arial.ttf

# Yolov8 src version: 8.0.198
git clone -b py37 https://gitee.com/jiongjiongli/yolov8.git ultralytics
cp /project/train/src_repo/yolov8n.pt /project/train/src_repo/ultralytics/

cd /project/train/src_repo
rm -rf /project/train/src_repo/dangerous_vehicle_det
git clone https://gitee.com/jiongjiongli/dangerous_vehicle_det.git

cp /project/train/src_repo/dangerous_vehicle_det/vehicle_det/start_train.sh /project/train/src_repo/

cp -r /project/train/src_repo/dangerous_vehicle_det/vehicle_det /project/train/src_repo/ultralytics/

cp /project/train/src_repo/dangerous_vehicle_det/vehicle_det/ji.py /project/ev_sdk/src/

rm -rf /project/train/models/*
rm -rf /project/train/tensorboard/*
rm -rf /project/train/log/*
rm -rf /project/train/result-graphs/*


