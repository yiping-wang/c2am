# echo "======= C2AM Train ========="
# mkdir -p /data/home/yipingwang/data/C2AM/
# mkdir -p /data/home/yipingwang/data/IRLabelC2AM/
# mkdir -p /data/home/yipingwang/data/SemSegC2AM/
# mkdir -p /data/home/yipingwang/data/GlobalCAM/
# rm /data/home/yipingwang/data/C2AM/*
# rm /data/home/yipingwang/data/IRLabelC2AM/*
# rm /data/home/yipingwang/data/SemSegC2AM/*
# python3 train_c2am.py             --config    ./cfg/c2am_exp.yml
# python3 make_cam.py               --config    ./cfg/c2am_exp.yml
# python3 eval_cam.py               --config    ./cfg/c2am_exp.yml --cam_eval_thres 0.16
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp.yml
# python3 train_irn.py              --config    ./cfg/c2am_exp.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp.yml
# echo "======================================="


echo "======= C2AM Reproduce ========="
mkdir -p /data/home/yipingwang/data/C2AMExp/
mkdir -p /data/home/yipingwang/data/IRLabelC2AMExp/
mkdir -p /data/home/yipingwang/data/SemSegC2AMExp/
mkdir -p /data/home/yipingwang/data/GlobalCAM/
rm /data/home/yipingwang/data/C2AMExp/*
rm /data/home/yipingwang/data/IRLabelC2AMExp/*
rm /data/home/yipingwang/data/SemSegC2AMExp/*
python3 make_cam.py               --config    ./cfg/c2am_exp_00.yml
python3 eval_cam.py               --config    ./cfg/c2am_exp_00.yml --cam_eval_thres 0.16
python3 eval_cam.py               --config    ./cfg/c2am_exp_00.yml --cam_eval_thres 0.18
python3 eval_cam.py               --config    ./cfg/c2am_exp_00.yml --cam_eval_thres 0.20
python3 eval_cam.py               --config    ./cfg/c2am_exp_00.yml --cam_eval_thres 0.22
python3 eval_cam.py               --config    ./cfg/c2am_exp_00.yml --cam_eval_thres 0.24
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp.yml
echo "======================================="