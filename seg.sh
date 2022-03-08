rm /data/home/yipingwang/data/CAMFdsi/*
rm /data/home/yipingwang/data/IRLabelFdsi/*
rm /data/home/yipingwang/data/SemSegFdsi/*

python3 make_cam_regular.py --config ./cfg/fdsi_exp15.yml
python3 cam_to_ir_label.py  --config ./cfg/fdsi_exp15.yml
python3 train_irn.py        --config ./cfg/ir_net_frontdoor.yml