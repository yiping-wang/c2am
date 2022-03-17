echo "======= Exp FDSI Dataset-wise 2 ========="
mkdir -p /data/home/yipingwang/data/CAMFdsi/
mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
mkdir -p /data/home/yipingwang/data/SemSegFdsi/
rm /data/home/yipingwang/data/CAMFdsi/*
rm /data/home/yipingwang/data/IRLabelFdsi/*
rm /data/home/yipingwang/data/SemSegFdsi/*
python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_02.yml
python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_02.yml
python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_02.yml
python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_02.yml
python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_02.yml
python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_02.yml
echo "======================================="

# echo "======= Exp ReCAM ====================="
# mkdir -p /data/home/yipingwang/data/CAMReCAM/
# mkdir -p /data/home/yipingwang/data/IRLabelReCAM/
# mkdir -p /data/home/yipingwang/data/SemSegReCAM/
# rm /data/home/yipingwang/data/CAMReCAM/*
# rm /data/home/yipingwang/data/IRLabelReCAM/*
# rm /data/home/yipingwang/data/SemSegReCAM/*
# python3 train_recam_fdsi.py      --config ./cfg/fdsi_expReCAM.yml
# python3 make_recam_regular.py    --config ./cfg/fdsi_expReCAM.yml
# python3 cam_to_ir_label.py       --config ./cfg/ir_net_expReCAM.yml
# python3 train_irn.py             --config ./cfg/ir_net_expReCAM.yml
# python3 make_sem_seg_labels.py   --config ./cfg/ir_net_expReCAM.yml
# python3 eval_sem_seg.py          --config ./cfg/ir_net_expReCAM.yml
# echo "======================================="

# echo "======= Exp FDSI Batch-wise ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_batchwise.py --config ./cfg/fdsi_batchwise_01.yml
# python3 make_cam_regular.py    --config  ./cfg/fdsi_batchwise_01.yml
# python3 cam_to_ir_label.py     --config  ./cfg/ir_net_batchwise_01.yml
# python3 train_irn.py           --config  ./cfg/ir_net_batchwise_01.yml
# python3 make_sem_seg_labels.py --config  ./cfg/ir_net_batchwise_01.yml
# python3 eval_sem_seg.py        --config  ./cfg/ir_net_batchwise_01.yml
# echo "========================"

# echo "======= Exp FDSI Sup Match 2 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_sup_match.py    --config ./cfg/fdsi_match_02.yml
# python3 make_cam_regular.py    --config    ./cfg/fdsi_match_02.yml
# python3 cam_to_ir_label.py     --config    ./cfg/ir_net_match_02.yml
# python3 train_irn.py           --config    ./cfg/ir_net_match_02.yml
# python3 make_sem_seg_labels.py --config    ./cfg/ir_net_match_02.yml
# python3 eval_sem_seg.py        --config    ./cfg/ir_net_match_02.yml
# echo "========================"

# echo "======= Exp FDSI Match 3 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_match.py    --config    ./cfg/fdsi_match_03.yml
# python3 make_cam_regular.py    --config    ./cfg/fdsi_match_03.yml
# python3 cam_to_ir_label.py     --config    ./cfg/ir_net_match_03.yml
# python3 train_irn.py           --config    ./cfg/ir_net_match_03.yml
# python3 make_sem_seg_labels.py --config    ./cfg/ir_net_match_03.yml
# python3 eval_sem_seg.py        --config    ./cfg/ir_net_match_03.yml
# echo "========================"

