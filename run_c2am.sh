echo "======= C2AM Train ========="
mkdir -p /u8/y3967wang/output/CAM_C2AM/
mkdir -p /u8/y3967wang/output/IRLabel_C2AM/
mkdir -p /u8/y3967wang/output/SemSeg_C2AM/
mkdir -p /u8/y3967wang/output/Global_CAM/
rm /u8/y3967wang/output/CAM_C2AM/*
rm /u8/y3967wang/output/IRLabel_C2AM/*
rm /u8/y3967wang/output/SemSeg_C2AM/*
# python3 train_c2am.py             --config    ./cfg/c2am_reproduce.yml
python3 make_cam.py               --config    ./cfg/c2am_reproduce.yml
python3 eval_cam.py               --config    ./cfg/c2am_reproduce.yml --cam_eval_thres 0.13
python3 eval_cam.py               --config    ./cfg/c2am_reproduce.yml --cam_eval_thres 0.14
python3 eval_cam.py               --config    ./cfg/c2am_reproduce.yml --cam_eval_thres 0.15
python3 eval_cam.py               --config    ./cfg/c2am_reproduce.yml --cam_eval_thres 0.16
python3 eval_cam.py               --config    ./cfg/c2am_reproduce.yml --cam_eval_thres 0.17
python3 eval_cam.py               --config    ./cfg/c2am_reproduce.yml --cam_eval_thres 0.18
python3 eval_cam.py               --config    ./cfg/c2am_reproduce.yml --cam_eval_thres 0.19
python3 eval_cam.py               --config    ./cfg/c2am_reproduce.yml --cam_eval_thres 0.20
python3 cam_to_ir_label.py        --config    ./cfg/c2am_reproduce.yml
python3 train_irn.py              --config    ./cfg/c2am_reproduce.yml
python3 make_sem_seg_labels.py    --config    ./cfg/c2am_reproduce.yml
python3 eval_sem_seg.py           --config    ./cfg/c2am_reproduce.yml
echo "======================================="