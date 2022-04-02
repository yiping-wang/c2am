echo "======= Exp C2AM Exp 05 Train ========="
mkdir -p /data/home/yipingwang/data/C2AM/
mkdir -p /data/home/yipingwang/data/IRLabelC2AM/
mkdir -p /data/home/yipingwang/data/SemSegC2AM/
mkdir -p /data/home/yipingwang/data/GlobalCAM/
rm /data/home/yipingwang/data/C2AM/*
rm /data/home/yipingwang/data/IRLabelC2AM/*
rm /data/home/yipingwang/data/SemSegC2AM/*
python3 make_cam.py               --config    ./cfg/c2am_exp_05.yml
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.10
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.12
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.14
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.16
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.18
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.20
echo "======================================="


echo "======= Exp C2AM Exp 06 Train ========="
python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_06.yml
python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_06.yml
python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_06.yml
echo "======================================="