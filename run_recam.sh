echo "======= Exp ReCAM with Front Door ========="
mkdir -p /data/home/yipingwang/data/CAMReCAM/
mkdir -p /data/home/yipingwang/data/IRLabelReCAM/
mkdir -p /data/home/yipingwang/data/SemSegReCAM/
rm /data/home/yipingwang/data/CAMReCAM/*
rm /data/home/yipingwang/data/IRLabelReCAM/*
rm /data/home/yipingwang/data/SemSegReCAM/*
python3 train_recam_fdsi.py      --config ./cfg/fdsi_expReCAM_fd.yml
python3 make_recam_regular.py    --config ./cfg/fdsi_expReCAM_fd.yml
python3 cam_to_ir_label.py       --config ./cfg/ir_net_expReCAM_fd.yml
python3 train_irn.py             --config ./cfg/ir_net_expReCAM_fd.yml
python3 make_sem_seg_labels.py   --config ./cfg/ir_net_expReCAM_fd.yml
python3 eval_sem_seg.py          --config ./cfg/ir_net_expReCAM_fd.yml
echo "========================"

echo "======= Exp ReCAM ========="
mkdir -p /data/home/yipingwang/data/CAMReCAM/
mkdir -p /data/home/yipingwang/data/IRLabelReCAM/
mkdir -p /data/home/yipingwang/data/SemSegReCAM/
rm /data/home/yipingwang/data/CAMReCAM/*
rm /data/home/yipingwang/data/IRLabelReCAM/*
rm /data/home/yipingwang/data/SemSegReCAM/*
# python3 train_recam_fdsi.py --config ./cfg/fdsi_expReCAM.yml
python3 make_recam_regular.py    --config ./cfg/fdsi_expReCAM.yml
python3 cam_to_ir_label.py       --config ./cfg/ir_net_expReCAM_defaultparams.yml
python3 train_irn.py             --config ./cfg/ir_net_expReCAM_defaultparams.yml
python3 make_sem_seg_labels.py   --config ./cfg/ir_net_expReCAM_defaultparams.yml
python3 eval_sem_seg.py          --config ./cfg/ir_net_expReCAM_defaultparams.yml
echo "========================"