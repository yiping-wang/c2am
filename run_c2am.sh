# echo "======= Exp C2AM Exp 05 Train ========="
# mkdir -p /data/home/yipingwang/data/C2AM/
# mkdir -p /data/home/yipingwang/data/IRLabelC2AM/
# mkdir -p /data/home/yipingwang/data/SemSegC2AM/
# mkdir -p /data/home/yipingwang/data/GlobalCAM/
# rm /data/home/yipingwang/data/C2AM/*
# rm /data/home/yipingwang/data/IRLabelC2AM/*
# rm /data/home/yipingwang/data/SemSegC2AM/*
# python3 make_cam.py               --config    ./cfg/c2am_exp_05.yml
# python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.12
# python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.14
# python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.16
# python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.18
# python3 eval_cam.py               --config    ./cfg/c2am_exp_05.yml --cam_eval_thres 0.20
# echo "======================================="

# echo "======= Exp C2AM Exp 06 Rep ========="
# rm /data/home/yipingwang/data/IRLabelC2AM/*
# rm /data/home/yipingwang/data/SemSegC2AM/*
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_06.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_06.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_06.yml
# echo "======================================="

echo "======= Exp C2AM Exp 07 Train ========="
rm /data/home/yipingwang/data/IRLabelC2AM/*
rm /data/home/yipingwang/data/SemSegC2AM/*
python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_07.yml
python3 train_irn.py              --config    ./cfg/c2am_exp_07.yml
python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_07.yml
python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_07.yml
echo "======================================="

echo "======= Exp C2AM Exp 08 Train ========="
rm /data/home/yipingwang/data/IRLabelC2AM/*
rm /data/home/yipingwang/data/SemSegC2AM/*
python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_08.yml
python3 train_irn.py              --config    ./cfg/c2am_exp_08.yml
python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_08.yml
python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_08.yml
echo "======================================="

# echo "======= Exp C2AM Exp 09 Train ========="
# rm /data/home/yipingwang/data/IRLabelC2AM/*
# rm /data/home/yipingwang/data/SemSegC2AM/*
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_09.yml
# python3 train_irn.py              --config    ./cfg/c2am_exp_09.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_09.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_09.yml
# echo "======================================="

# echo "======= Exp C2AM Exp 10 Train ========="
# rm /data/home/yipingwang/data/IRLabelC2AM/*
# rm /data/home/yipingwang/data/SemSegC2AM/*
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_10.yml
# python3 train_irn.py              --config    ./cfg/c2am_exp_10.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_10.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_10.yml
# echo "======================================="

# echo "======= Exp C2AM Exp 11 Train ========="
# rm /data/home/yipingwang/data/IRLabelC2AM/*
# rm /data/home/yipingwang/data/SemSegC2AM/*
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_11.yml
# python3 train_irn.py              --config    ./cfg/c2am_exp_11.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_11.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_11.yml
# echo "======================================="

# echo "======= Exp C2AM Exp 04 Train ========="
# mkdir -p /data/home/yipingwang/data/C2AM/
# mkdir -p /data/home/yipingwang/data/IRLabelC2AM/
# mkdir -p /data/home/yipingwang/data/SemSegC2AM/
# mkdir -p /data/home/yipingwang/data/GlobalCAM/
# rm /data/home/yipingwang/data/C2AM/*
# rm /data/home/yipingwang/data/IRLabelC2AM/*
# rm /data/home/yipingwang/data/SemSegC2AM/*
# python3 train_c2am.py             --config    ./cfg/c2am_exp_04.yml
# python3 make_cam.py               --config    ./cfg/c2am_exp_04.yml
# python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.10
# python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.12
# python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.14
# python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.16
# python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.18
# python3 eval_cam.py               --config    ./cfg/c2am_exp_04.yml --cam_eval_thres 0.20
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_04.yml
# python3 train_irn.py              --config    ./cfg/c2am_exp_04.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_04.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_04.yml
# echo "======================================="

# echo "======= C2AM Reproduce ========="
# mkdir -p /data/home/yipingwang/data/C2AM/
# mkdir -p /data/home/yipingwang/data/IRLabelC2AM/
# mkdir -p /data/home/yipingwang/data/SemSegC2AM/
# mkdir -p /data/home/yipingwang/data/GlobalCAM/
# rm /data/home/yipingwang/data/C2AM/*
# rm /data/home/yipingwang/data/IRLabelC2AM/*
# rm /data/home/yipingwang/data/SemSegC2AM/*
# # python3 train_c2am.py             --config    ./cfg/c2am_exp_01.yml
# python3 make_cam.py               --config    ./cfg/c2am_exp_01.yml
# python3 eval_cam.py               --config     ./cfg/c2am_exp_02.yml --cam_eval_thres 0.15
# python3 eval_cam.py               --config     ./cfg/c2am_exp_02.yml --cam_eval_thres 0.17
# python3 eval_cam.py               --config     ./cfg/c2am_exp_02.yml --cam_eval_thres 0.20
# python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_01.yml
# # python3 train_irn.py              --config    ./cfg/c2am_exp_01.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_01.yml
# python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_01.yml
# echo "======================================="