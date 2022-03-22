echo "======= Exp C2AM Exp 04 ========="
mkdir -p /data/home/yipingwang/data/C2AM/
mkdir -p /data/home/yipingwang/data/IRLabelC2AM/
mkdir -p /data/home/yipingwang/data/SemSegC2AM/
mkdir -p /data/home/yipingwang/data/GlobalCAM/
rm /data/home/yipingwang/data/C2AM/*
rm /data/home/yipingwang/data/IRLabelC2AM/*
rm /data/home/yipingwang/data/SemSegC2AM/*
python3 train_c2am.py             --config    ./cfg/c2am_exp_04.yml
python3 make_cam.py               --config    ./cfg/c2am_exp_04.yml
python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.15
python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.16
python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.17
python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.18
python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.19
python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.20
python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.21
python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.22
python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.23
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_04.yml
# python3 train_irn.py              --config    ./cfg/c2am_exp_04.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_04.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_04.yml
echo "======================================="

echo "======= Exp C2AM Exp 05 ========="
mkdir -p /data/home/yipingwang/data/C2AM/
mkdir -p /data/home/yipingwang/data/IRLabelC2AM/
mkdir -p /data/home/yipingwang/data/SemSegC2AM/
mkdir -p /data/home/yipingwang/data/GlobalCAM/
rm /data/home/yipingwang/data/C2AM/*
rm /data/home/yipingwang/data/IRLabelC2AM/*
rm /data/home/yipingwang/data/SemSegC2AM/*
python3 train_c2am.py             --config    ./cfg/c2am_exp_05.yml
python3 make_cam.py               --config    ./cfg/c2am_exp_05.yml
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.15
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.16
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.17
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.18
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.19
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.20
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.21
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.22
python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.23
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_05.yml
# python3 train_irn.py              --config    ./cfg/c2am_exp_05.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_05.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_05.yml
echo "======================================="