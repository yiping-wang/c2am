echo "======= Exp C2AM Exp 02 Train ========="
mkdir -p /data/home/yipingwang/data/C2AM/
mkdir -p /data/home/yipingwang/data/IRLabelC2AM/
mkdir -p /data/home/yipingwang/data/SemSegC2AM/
mkdir -p /data/home/yipingwang/data/GlobalCAM/
# rm /data/home/yipingwang/data/C2AM/*
# rm /data/home/yipingwang/data/IRLabelC2AM/*
# rm /data/home/yipingwang/data/SemSegC2AM/*
# python3 train_c2am.py             --config    ./cfg/c2am_exp_02.yml
# python3 make_cam.py               --config    ./cfg/c2am_exp_01.yml
python3 cam_to_ir_label.py        --config    ./cfg/c2am_exp_02.yml
python3 train_irn.py              --config    ./cfg/c2am_exp_02.yml
python3 make_sem_seg_labels.py    --config    ./cfg/c2am_exp_02.yml
python3 eval_sem_seg.py           --config    ./cfg/c2am_exp_02.yml
echo "======================================="