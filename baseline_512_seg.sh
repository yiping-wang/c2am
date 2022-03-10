rm /data/home/yipingwang/data/CAMBaseline/*
rm /data/home/yipingwang/data/IRLabelBaseline/*
rm /data/home/yipingwang/data/SemSegBaseline/*

python3 make_cam_regular.py    --config ./cfg/baseline_512.yml
python3 cam_to_ir_label.py     --config ./cfg/ir_net_baseline_512.yml
python3 train_irn.py           --config ./cfg/ir_net_baseline_512.yml
python3 make_sem_seg_labels.py --config ./cfg/ir_net_baseline_512.yml
python3 eval_sem_seg.py        --config ./cfg/ir_net_baseline_512.yml