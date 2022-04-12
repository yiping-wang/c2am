echo "======= ReCAM Exp 01 ========="
mkdir -p /data/home/yipingwang/data/CAMReCAM/
mkdir -p /data/home/yipingwang/data/IRLabelReCAM/
mkdir -p /data/home/yipingwang/data/SemSegReCAM/
rm /data/home/yipingwang/data/CAMReCAM/*
rm /data/home/yipingwang/data/IRLabelReCAM/*
rm /data/home/yipingwang/data/SemSegReCAM/*
python3 train_recam.py           --config ./cfg/recam/recam_exp_01.yml
python3 make_recam.py            --config ./cfg/recam/recam_exp_01.yml
python3 eval_cam.py              --config ./cfg/recam/recam_exp_01.yml --cam_eval_thres 0.13
python3 eval_cam.py              --config ./cfg/recam/recam_exp_01.yml --cam_eval_thres 0.15
python3 eval_cam.py              --config ./cfg/recam/recam_exp_01.yml --cam_eval_thres 0.17
python3 eval_cam.py              --config ./cfg/recam/recam_exp_01.yml --cam_eval_thres 0.19
python3 eval_cam.py              --config ./cfg/recam/recam_exp_01.yml --cam_eval_thres 0.21
# python3 cam_to_ir_label.py       --config ./cfg/recam/recam_exp_01.yml
# python3 train_irn.py             --config ./cfg/recam/recam_exp_01.yml
# python3 make_sem_seg_labels.py   --config ./cfg/recam/recam_exp_01.yml
# python3 eval_sem_seg.py          --config ./cfg/recam/recam_exp_01.yml
echo "========================"