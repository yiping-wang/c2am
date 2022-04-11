echo "======= CAM Constant Train ========="
mkdir -p /data/home/yipingwang/data/C2AMAblation/
mkdir -p /data/home/yipingwang/data/IRLabelC2AMAblation/
mkdir -p /data/home/yipingwang/data/SemSegC2AMAblation/
rm /data/home/yipingwang/data/C2AMAblation/*
rm /data/home/yipingwang/data/IRLabelC2AMAblation/*
rm /data/home/yipingwang/data/SemSegC2AMAblation/*
python3 train_c3am.py             --config    ./cfg/cam_constant.yml
python3 make_cam.py               --config    ./cfg/cam_constant.yml
python3 eval_cam.py               --config    ./cfg/cam_constant.yml --cam_eval_thres 0.13
python3 eval_cam.py               --config    ./cfg/cam_constant.yml --cam_eval_thres 0.14
python3 eval_cam.py               --config    ./cfg/cam_constant.yml --cam_eval_thres 0.15
python3 eval_cam.py               --config    ./cfg/cam_constant.yml --cam_eval_thres 0.16
python3 eval_cam.py               --config    ./cfg/cam_constant.yml --cam_eval_thres 0.17
python3 eval_cam.py               --config    ./cfg/cam_constant.yml --cam_eval_thres 0.18
python3 eval_cam.py               --config    ./cfg/cam_constant.yml --cam_eval_thres 0.19
python3 eval_cam.py               --config    ./cfg/cam_constant.yml --cam_eval_thres 0.20
python3 cam_to_ir_label.py        --config    ./cfg/cam_constant.yml
python3 train_irn.py              --config    ./cfg/cam_constant.yml
python3 make_sem_seg_labels.py    --config    ./cfg/cam_constant.yml
python3 eval_sem_seg.py           --config    ./cfg/cam_constant.yml
echo "======================================="
