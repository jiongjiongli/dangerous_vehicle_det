# CVMart dangerous vehicle detect
# 1. Prepare CVMart Env
```bash
!rm /project/train/src_repo/cvmart_env.sh

!wget -P /project/train/src_repo/ https://gitee.com/jiongjiongli/dangerous_vehicle_det/raw/main/vehicle_det/cvmart_env.sh

!bash /project/train/src_repo/cvmart_env.sh
```



# 2. Train
```bash
!bash /project/train/src_repo/start_train.sh
```



# 3. Test

```
!python /project/train/src_repo/ji.py
```

