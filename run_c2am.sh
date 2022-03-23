echo "======= Exp C2AM Exp 03 Train ========="
mkdir -p /data/home/yipingwang/data/C2AM/
mkdir -p /data/home/yipingwang/data/IRLabelC2AM/
mkdir -p /data/home/yipingwang/data/SemSegC2AM/
mkdir -p /data/home/yipingwang/data/GlobalCAM/
rm /data/home/yipingwang/data/C2AM/*
rm /data/home/yipingwang/data/IRLabelC2AM/*
rm /data/home/yipingwang/data/SemSegC2AM/*
python3 train_c2am.py             --config    ./cfg/c2am_exp_03.yml
python3 make_cam.py               --config    ./cfg/c2am_exp_03.yml
python3 eval_cam.py               --config    ./cfg/c2am_exp_03.yml --cam_eval_thres 0.15
python3 eval_cam.py               --config    ./cfg/c2am_exp_03.yml --cam_eval_thres 0.17
python3 eval_cam.py               --config    ./cfg/c2am_exp_03.yml --cam_eval_thres 0.19
python3 eval_cam.py               --config    ./cfg/c2am_exp_03.yml --cam_eval_thres 0.21
python3 eval_cam.py               --config    ./cfg/c2am_exp_03.yml --cam_eval_thres 0.23
python3 eval_cam.py               --config    ./cfg/c2am_exp_03.yml --cam_eval_thres 0.25
python3 eval_cam.py               --config    ./cfg/c2am_exp_03.yml --cam_eval_thres 0.27
echo "======================================="

echo "======= ReCAM Exp 01 ========="
mkdir -p /data/home/yipingwang/data/CAMReCAM/
mkdir -p /data/home/yipingwang/data/IRLabelReCAM/
mkdir -p /data/home/yipingwang/data/SemSegReCAM/
rm /data/home/yipingwang/data/CAMReCAM/*
rm /data/home/yipingwang/data/IRLabelReCAM/*
rm /data/home/yipingwang/data/SemSegReCAM/*
python3 train_recam.py           --config ./cfg/recam/recam_exp_01.yml
python3 make_recam.py            --config ./cfg/recam/recam_exp_01.yml
python3 eval_cam.py              --config ./cfg/recam/recam_exp_01.yml --cam_eval_thres 0.21
python3 cam_to_ir_label.py       --config ./cfg/recam/recam_exp_01.yml
python3 train_irn.py             --config ./cfg/recam/recam_exp_01.yml
python3 make_sem_seg_labels.py   --config ./cfg/recam/recam_exp_01.yml
python3 eval_sem_seg.py          --config ./cfg/recam/recam_exp_01.yml
echo "========================"

echo "======= ReCAM Exp 02 ========="
mkdir -p /data/home/yipingwang/data/CAMReCAM/
mkdir -p /data/home/yipingwang/data/IRLabelReCAM/
mkdir -p /data/home/yipingwang/data/SemSegReCAM/
rm /data/home/yipingwang/data/CAMReCAM/*
rm /data/home/yipingwang/data/IRLabelReCAM/*
rm /data/home/yipingwang/data/SemSegReCAM/*
python3 train_recam.py           --config ./cfg/recam/recam_exp_02.yml
python3 make_recam.py            --config ./cfg/recam/recam_exp_02.yml
python3 eval_cam.py              --config ./cfg/recam/recam_exp_02.yml --cam_eval_thres 0.21
python3 cam_to_ir_label.py       --config ./cfg/recam/recam_exp_02.yml
python3 train_irn.py             --config ./cfg/recam/recam_exp_02.yml
python3 make_sem_seg_labels.py   --config ./cfg/recam/recam_exp_02.yml
python3 eval_sem_seg.py          --config ./cfg/recam/recam_exp_02.yml
echo "========================"

echo "======= ReCAM Exp 03 ========="
mkdir -p /data/home/yipingwang/data/CAMReCAM/
mkdir -p /data/home/yipingwang/data/IRLabelReCAM/
mkdir -p /data/home/yipingwang/data/SemSegReCAM/
rm /data/home/yipingwang/data/CAMReCAM/*
rm /data/home/yipingwang/data/IRLabelReCAM/*
rm /data/home/yipingwang/data/SemSegReCAM/*
python3 train_recam.py           --config ./cfg/recam/recam_exp_03.yml
python3 make_recam.py            --config ./cfg/recam/recam_exp_03.yml
python3 eval_cam.py              --config ./cfg/recam/recam_exp_03.yml --cam_eval_thres 0.21
python3 cam_to_ir_label.py       --config ./cfg/recam/recam_exp_03.yml
python3 train_irn.py             --config ./cfg/recam/recam_exp_03.yml
python3 make_sem_seg_labels.py   --config ./cfg/recam/recam_exp_03.yml
python3 eval_sem_seg.py          --config ./cfg/recam/recam_exp_03.yml
echo "========================"