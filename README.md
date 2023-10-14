# CVMart dangerous vehicle detect
# 1. Prepare CVMart Env
```bash
!rm /project/train/src_repo/cvmart_env.sh

!wget -P /project/train/src_repo/ https://raw.githubusercontent.com/jiongjiongli/dangerous_vehicle_det/main/vehicle_det/cvmart_env.sh

!bash /project/train/src_repo/cvmart_env.sh
```



# 2. Train
```bash
!bash /project/train/src_repo/start_train.sh
```

