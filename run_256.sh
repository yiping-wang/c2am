# echo "================== Baseline ==========================="
# rm /data/home/yipingwang/data/CAMBaseline/*
# rm /data/home/yipingwang/data/IRLabelBaseline/*
# rm /data/home/yipingwang/data/SemSegBaseline/*
# python3 make_cam_regular.py    --config ./cfg/baseline_256.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_baseline_256.yml
# python3 train_irn.py           --config ./cfg/ir_net_baseline_256.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_baseline_256.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_baseline_256.yml
# echo "=========================================================="


echo "================== Exp 15 ================================="
rm /data/home/yipingwang/data/CAMFdsiExp15/*
rm /data/home/yipingwang/data/IRLabelFdsiExp15/*
rm /data/home/yipingwang/data/SemSegFdsiExp15/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp15.yml
python3 make_cam_regular.py    --config ./cfg/fdsi_exp15.yml
python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp15.yml
python3 train_irn.py           --config ./cfg/ir_net_exp15.yml
python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp15.yml
python3 eval_sem_seg.py        --config ./cfg/ir_net_exp15.yml
echo "=========================================================="


# echo "================== Exp 1 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp1.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp1.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp1.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp1.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp1.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp1.yml
# echo "=========================================================="

# echo "================== Exp 2 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp2.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp2.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp2.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp2.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp2.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp2.yml
# echo "=========================================================="

# echo "================== Exp 3 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp3.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp3.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp3.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp3.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp3.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp3.yml
# echo "=========================================================="

# echo "================== Exp 4 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp4.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp4.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp4.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp4.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp4.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp4.yml
# echo "=========================================================="

# echo "================== Exp 5 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp5.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp5.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp5.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp5.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp5.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp5.yml
# echo "=========================================================="

# echo "================== Exp 6 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp6.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp6.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp6.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp6.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp6.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp6.yml
# echo "=========================================================="

# echo "================== Exp 7 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp7.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp7.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp7.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp7.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp7.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp7.yml
# echo "=========================================================="

# echo "================== Exp 8 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp8.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp8.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp8.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp8.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp8.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp8.yml
# echo "=========================================================="


# echo "================== Exp 9 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp9.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp9.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp9.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp9.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp9.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp9.yml
# echo "=========================================================="

# echo "================== Exp 10 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp10.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp10.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp10.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp10.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp10.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp10.yml
# echo "=========================================================="


# echo "================== Exp 11 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp11.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp11.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp11.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp11.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp11.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp11.yml
# echo "=========================================================="

# echo "================== Exp 12 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp12.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp12.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp12.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp12.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp12.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp12.yml
# echo "=========================================================="

# echo "================== Exp 13 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp13.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp13.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp13.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp13.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp13.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp13.yml
# echo "==========================================================" 


# echo "================== Exp 14 ================================="
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi.py --config ./cfg/fdsi_exp14.yml
# python3 make_cam_regular.py    --config ./cfg/fdsi_exp14.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_exp14.yml
# python3 train_irn.py           --config ./cfg/ir_net_exp14.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_exp14.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_exp14.yml
# echo "==========================================================" 


# echo "================== Baseline E10 ==========================="
# rm /data/home/yipingwang/data/CAMBaseline/*
# rm /data/home/yipingwang/data/IRLabelBaseline/*
# rm /data/home/yipingwang/data/SemSegBaseline/*
# python3 make_cam_regular.py    --config ./cfg/baseline_256_e10.yml
# python3 cam_to_ir_label.py     --config ./cfg/ir_net_baseline_256_e10.yml
# python3 train_irn.py           --config ./cfg/ir_net_baseline_256_e10.yml
# python3 make_sem_seg_labels.py --config ./cfg/ir_net_baseline_256_e10.yml
# python3 eval_sem_seg.py        --config ./cfg/ir_net_baseline_256_e10.yml
# echo "=========================================================="