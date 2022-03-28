echo "======= Baseline ========="
mkdir -p /data/home/yipingwang/data/CAMBaseline/
mkdir -p /data/home/yipingwang/data/IRLabelBaseline/
mkdir -p /data/home/yipingwang/data/SemSegBaseline/
rm /data/home/yipingwang/data/CAMBaseline/*
rm /data/home/yipingwang/data/IRLabelBaseline/*
rm /data/home/yipingwang/data/SemSegBaseline/*
#python3 train_baseline.py         --config    ./cfg/baseline/baseline_512.yml
python3 make_cam.py                --config    ./cfg/baseline/baseline_512.yml
python3 eval_cam.py               --config     ./cfg/baseline/baseline_512.yml --cam_eval_thres 0.03
python3 eval_cam.py               --config     ./cfg/baseline/baseline_512.yml --cam_eval_thres 0.05
python3 eval_cam.py               --config     ./cfg/baseline/baseline_512.yml --cam_eval_thres 0.08
python3 eval_cam.py               --config     ./cfg/baseline/baseline_512.yml --cam_eval_thres 0.10
python3 eval_cam.py               --config     ./cfg/baseline/baseline_512.yml --cam_eval_thres 0.15
python3 eval_cam.py               --config     ./cfg/baseline/baseline_512.yml --cam_eval_thres 0.20
python3 cam_to_ir_label.py         --config    ./cfg/baseline/baseline_512.yml
python3 train_irn.py               --config    ./cfg/baseline/baseline_512.yml
python3 make_sem_seg_labels.py     --config    ./cfg/baseline/baseline_512.yml
python3 eval_sem_seg.py            --config    ./cfg/baseline/baseline_512.yml
echo "======================================="