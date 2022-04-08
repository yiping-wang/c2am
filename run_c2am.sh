echo "======= C2AM Train ========="
mkdir -p /data/home/yipingwang/data/C2AMAblation/
mkdir -p /data/home/yipingwang/data/IRLabelC2AMAblation/
mkdir -p /data/home/yipingwang/data/SemSegC2AMAblation/
mkdir -p /data/home/yipingwang/data/GlobalCAM/
rm /data/home/yipingwang/data/C2AMAblation/*
rm /data/home/yipingwang/data/IRLabelC2AMAblation/*
rm /data/home/yipingwang/data/SemSegC2AMAblation/*
python3 train_c2am.py             --config    ./cfg/c2am_exp_07.yml
python3 make_cam.py               --config    ./cfg/c2am_exp_07.yml
python3 eval_cam.py               --config    ./cfg/c2am_exp_07.yml --cam_eval_thres 0.13
python3 eval_cam.py               --config    ./cfg/c2am_exp_07.yml --cam_eval_thres 0.14
python3 eval_cam.py               --config    ./cfg/c2am_exp_07.yml --cam_eval_thres 0.15
python3 eval_cam.py               --config    ./cfg/c2am_exp_07.yml --cam_eval_thres 0.16
python3 eval_cam.py               --config    ./cfg/c2am_exp_07.yml --cam_eval_thres 0.17
python3 eval_cam.py               --config    ./cfg/c2am_exp_07.yml --cam_eval_thres 0.18
python3 eval_cam.py               --config    ./cfg/c2am_exp_07.yml --cam_eval_thres 0.19
python3 eval_cam.py               --config    ./cfg/c2am_exp_07.yml --cam_eval_thres 0.20
python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_07.yml
python3 train_irn.py              --config    ./cfg/c2am_exp_07.yml
python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_07.yml
python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_07.yml
echo "======================================="


# echo "======= C2AM Reproduce ========="
# mkdir -p /data/home/yipingwang/data/C2AMExp/
# mkdir -p /data/home/yipingwang/data/IRLabelC2AMExp/
# mkdir -p /data/home/yipingwang/data/SemSegC2AMExp/
# mkdir -p /data/home/yipingwang/data/GlobalCAM/
# rm /data/home/yipingwang/data/C2AMExp/*
# rm /data/home/yipingwang/data/IRLabelC2AMExp/*
# rm /data/home/yipingwang/data/SemSegC2AMExp/*
# python3 make_cam.py               --config    ./cfg/c2am_exp_00.yml
# python3 eval_cam.py               --config    ./cfg/c2am_exp_00.yml --cam_eval_thres 0.16
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp.yml
# echo "======================================="