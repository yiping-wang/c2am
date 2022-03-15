echo "======= Exp 36 ========="
rm /data/home/yipingwang/data/CAMFdsi/*
rm /data/home/yipingwang/data/IRLabelFdsi/*
rm /data/home/yipingwang/data/SemSegFdsi/*
python3 train_fdsi.py --config ./cfg/fdsi_exp36.yml
python3 make_cam_regular.py    --config ./cfg/fdsi_exp36.yml
python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp36.yml
python3 train_irn.py           --config ./cfg/ir_net_exp36.yml
python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp36.yml
python3 eval_sem_seg.py        --config ./cfg/ir_net_exp36.yml
echo "========================"

# echo "======= Exp 35 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp35.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp35.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp35.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp35.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp35.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp35.yml
# echo "========================"

# echo "======= Exp 15 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp15.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp15.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp15.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp15.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp15.yml
# echo "========================"


# echo "======= Exp 32 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp32.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp32.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp32.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp32.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp32.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp32.yml
# echo "========================"


# echo "======= Exp 33 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp33.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp33.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp33.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp33.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp33.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp33.yml
# echo "========================"


# echo "======= Exp 25 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp25.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp25.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp25.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp25.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp25.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp25.yml
# echo "========================"


# echo "======= Exp 26 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp26.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp26.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp26.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp26.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp26.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp26.yml
# echo "========================"

# echo "======= Exp 27 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp27.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp27.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp27.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp27.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp27.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp27.yml
# echo "========================"

# echo "======= Exp 28 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp28.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp28.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp28.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp28.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp28.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp28.yml
# echo "========================"

# echo "======= Exp 29 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp29.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp29.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp29.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp29.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp29.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp29.yml
# echo "========================"


# echo "======= Exp 19 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp19.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp19.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp19.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp19.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp19.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp19.yml
# echo "========================"

# echo "======= Exp 21 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp21.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp21.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp21.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp21.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp21.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp21.yml
# echo "========================"



# echo "======= Exp 18 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp18.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp18.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp18.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp18.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp18.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp18.yml
# echo "========================"

# echo "======= Exp 20 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp20.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp20.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp20.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp20.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp20.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp20.yml
# echo "========================"

# echo "======= Exp 22 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp22.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp22.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp22.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp22.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp22.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp22.yml
# echo "========================"


# echo "======= Exp 23 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp23.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp23.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp23.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp23.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp23.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp23.yml
# echo "========================"

# echo "======= Exp 24 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp24.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp24.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp24.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp24.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp24.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp24.yml
# echo "========================"

# echo "======= Exp 30 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp30.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp30.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp30.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp30.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp30.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp30.yml
# echo "========================"

# echo "======= Exp 31 ========="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp31.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp31.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp31.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp31.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp31.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp31.yml
# echo "========================"