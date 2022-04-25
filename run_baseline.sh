echo "======= Baseline ========="
mkdir -p /data/home/yipingwang/data/CAMBaseline/
mkdir -p /data/home/yipingwang/data/IRLabelBaseline/
mkdir -p /data/home/yipingwang/data/SemSegBaseline/
rm /data/home/yipingwang/data/CAMBaseline/*
rm /data/home/yipingwang/data/IRLabelBaseline/*
rm /data/home/yipingwang/data/SemSegBaseline/*
# python3 train_baseline.py       --config    ./cfg/baseline/baseline.yml
python3 make_cam.py               --config    ./cfg/baseline/baseline.yml
python3 eval_cam.py               --config    ./cfg/baseline/baseline.yml --cam_eval_thres 0.10
# python3 cam_to_ir_label.py      --config    ./cfg/baseline/baseline.yml
# python3 train_irn.py            --config    ./cfg/baseline/baseline.yml
python3 make_sem_seg_labels.py    --config    ./cfg/baseline/baseline.yml
python3 eval_sem_seg.py           --config    ./cfg/baseline/baseline.yml
echo "======================================="