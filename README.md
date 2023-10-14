# dangerous_vehicle_det
Dangerous vehicle detect.

# Prepare CVMart Env
rm /project/train/src_repo/cvmart_env.sh

!wget -P /project/train/src_repo/ https://gitee.com/jiongjiongli/dangerous_vehicle_det/blob/main/vehicle_det/cvmart_env.sh

!bash /project/train/src_repo/cvmart_env.sh

# Train
!bash /project/train/src_repo/start_train.sh
