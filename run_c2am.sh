echo "======= Exp C2AM Exp 01 ========="
mkdir -p /data/home/yipingwang/data/C2AM/
mkdir -p /data/home/yipingwang/data/IRLabelC2AM/
mkdir -p /data/home/yipingwang/data/SemSegC2AM/
mkdir -p /data/home/yipingwang/data/GlobalCAM/
rm /data/home/yipingwang/data/C2AM/*
rm /data/home/yipingwang/data/IRLabelC2AM/*
rm /data/home/yipingwang/data/SemSegC2AM/*
python3 train_c2am.py             --config    ./cfg/c2am_exp_01.yml
python3 make_cam.py               --config    ./cfg/c2am_exp_01.yml
python3 cam_to_ir_label.py        --config    ./cfg/irn_c2am_exp_01.yml
python3 train_irn.py              --config    ./cfg/irn_c2am_exp_01.yml
python3 make_sem_seg_labels.py    --config    ./cfg/irn_c2am_exp_01.yml
python3 eval_sem_seg.py           --config    ./cfg/irn_c2am_exp_01.yml
echo "======================================="


# echo "======= Exp FDSI Dataset-wise 26 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_26.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_26.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_26.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_26.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_26.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_26.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 27 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_27.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_27.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_27.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_27.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_27.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_27.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 26 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_26.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_26.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_26.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_26.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_26.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_26.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 25 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_25.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_25.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_25.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_25.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_25.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_25.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 24 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_24.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_24.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_24.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_24.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_24.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_24.yml
# echo "======================================="

# echo "======= Exp Tune IRNet on C2AM 01 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_13.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_c2am_01.yml
# python3 train_irn.py              --config    ./cfg/ir_net_c2am_01.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_c2am_01.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_c2am_01.yml
# echo "======================================="

# echo "======= Exp Tune IRNet on C2AM 02 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_13.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_c2am_02.yml
# python3 train_irn.py              --config    ./cfg/ir_net_c2am_02.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_c2am_02.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_c2am_02.yml
# echo "======================================="

# echo "======= Exp Tune IRNet on C2AM 03 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_13.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_c2am_03.yml
# python3 train_irn.py              --config    ./cfg/ir_net_c2am_03.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_c2am_03.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_c2am_03.yml
# echo "======================================="

# echo "======= Exp Tune IRNet on C2AM 04 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_13.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_c2am_04.yml
# python3 train_irn.py              --config    ./cfg/ir_net_c2am_04.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_c2am_04.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_c2am_04.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 22 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_22.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_22.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_22.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_22.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_22.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_22.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 23 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_23.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_23.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_23.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_23.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_23.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_23.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 21 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_21.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_21.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_21.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_21.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_21.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_21.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 20 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_20.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_20.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_20.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_20.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_20.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_20.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 18 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_18.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_18.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_18.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_18.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_18.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_18.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 19 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_19.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_19.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_19.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_19.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_19.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_19.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 14 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_14.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_14.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_14.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_14.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_14.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_14.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 15 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_15.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_15.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_15.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_15.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_15.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_15.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 16 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_16.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_16.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_16.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_16.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_16.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_16.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 17 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_17.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_17.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_17.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_17.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_17.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_17.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 13 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_13.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_13.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_13.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_13.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_13.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_13.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 12 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_12.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_12.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_12.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_12.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_12.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_12.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 11 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_11.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_11.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_11.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_11.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_11.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_11.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 8 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_08.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_08.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_08.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_08.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_08.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_08.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 9 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_09.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_09.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_09.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_09.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_09.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_09.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 7 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_07.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_07.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_07.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_07.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_07.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_07.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 5 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_05.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_05.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_05.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_05.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_05.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_05.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 6 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_06.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_06.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_06.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_06.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_06.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_06.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise ReOpt 1 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_dswise_reopt.py --config   ./cfg/fdsi_dswise_reopt_01.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_dswise_reopt_01.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_dswise_reopt_01.yml
# python3 train_irn.py              --config    ./cfg/ir_net_dswise_reopt_01.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_dswise_reopt_01.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_dswise_reopt_01.yml
# echo "======================================="


# echo "======= Exp FDSI Dataset-wise 3 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_03.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_03.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_03.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_03.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_03.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_03.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 4 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_04.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_04.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_04.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_04.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_04.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_04.yml
# echo "======================================="


# echo "======= Exp FDSI Match 3 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_match.py    --config ./cfg/fdsi_match_03.yml

# echo "======= Exp FDSI Dataset-wise 3 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_03.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_03.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_03.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_03.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_03.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_03.yml
# echo "======================================="

# echo "======= Exp FDSI Dataset-wise 2 ========="
# mkdir -p /data/home/yipingwang/data/CAMFdsi/
# mkdir -p /data/home/yipingwang/data/IRLabelFdsi/
# mkdir -p /data/home/yipingwang/data/SemSegFdsi/
# rm /data/home/yipingwang/data/CAMFdsi/*
# rm /data/home/yipingwang/data/IRLabelFdsi/*
# rm /data/home/yipingwang/data/SemSegFdsi/*
# python3 train_fdsi_datasetwise.py --config    ./cfg/fdsi_datasetwise_02.yml
# python3 make_cam_regular.py       --config    ./cfg/fdsi_datasetwise_02.yml
# python3 cam_to_ir_label.py        --config    ./cfg/ir_net_datasetwise_02.yml
# python3 train_irn.py              --config    ./cfg/ir_net_datasetwise_02.yml
# python3 make_sem_seg_labels.py    --config    ./cfg/ir_net_datasetwise_02.yml
# python3 eval_sem_seg.py           --config    ./cfg/ir_net_datasetwise_02.yml
# echo "======================================="

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

