python3 train_fdsi.py --config ./cfg/fdsi_exp18.yml

rm /data/home/yipingwang/data/CAMFdsi/*
rm /data/home/yipingwang/data/IRLabelFdsi/*
rm /data/home/yipingwang/data/SemSegFdsi/*
python3 make_cam_regular.py    --config ./cfg/fdsi_exp18.yml
python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp18.yml
python3 train_irn.py           --config ./cfg/ir_net_exp18.yml
python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp18.yml
python3 eval_sem_seg.py        --config ./cfg/ir_net_exp18.yml


python3 train_fdsi.py --config ./cfg/fdsi_exp19.yml

rm /data/home/yipingwang/data/CAMFdsi/*
rm /data/home/yipingwang/data/IRLabelFdsi/*
rm /data/home/yipingwang/data/SemSegFdsi/*
python3 make_cam_regular.py    --config ./cfg/fdsi_exp19.yml
python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp19.yml
python3 train_irn.py           --config ./cfg/ir_net_exp19.yml
python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp19.yml
python3 eval_sem_seg.py        --config ./cfg/ir_net_exp19.yml


python3 train_fdsi.py --config ./cfg/fdsi_exp20.yml

rm /data/home/yipingwang/data/CAMFdsi/*
rm /data/home/yipingwang/data/IRLabelFdsi/*
rm /data/home/yipingwang/data/SemSegFdsi/*
python3 make_cam_regular.py    --config ./cfg/fdsi_exp20.yml
python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp20.yml
python3 train_irn.py           --config ./cfg/ir_net_exp20.yml
python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp20.yml
python3 eval_sem_seg.py        --config ./cfg/ir_net_exp20.yml


python3 train_fdsi.py --config ./cfg/fdsi_exp21.yml

rm /data/home/yipingwang/data/CAMFdsi/*
rm /data/home/yipingwang/data/IRLabelFdsi/*
rm /data/home/yipingwang/data/SemSegFdsi/*
python3 make_cam_regular.py    --config ./cfg/fdsi_exp21.yml
python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp21.yml
python3 train_irn.py           --config ./cfg/ir_net_exp21.yml
python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp21.yml
python3 eval_sem_seg.py        --config ./cfg/ir_net_exp21.yml
