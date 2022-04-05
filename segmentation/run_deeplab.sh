python main.py train --config-path configs/voc12.yaml                           
python main.py test  --config-path ./configs/voc12.yaml --model-path ./data/models/voc12/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth              
cp ./data/models/voc12/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth    ./data/models/voc12/deeplabv2_resnet101_msc/train_aug/checkpoint_exp02.pth    
python main.py crf   --config-path ./configs/voc12.yaml