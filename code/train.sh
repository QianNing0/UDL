# !/bin/sh

## train step1
python main.py --model edsr --scale 4 --batch_size 64 --save ../experiment/X4/EDSR_edsr_s_x4_step1 --pre_train . --pre_train_step1 . --repeat 6 --stage step1 --ext sep
## train step2
python main.py --model edsr_two --scale 4 --batch_size 64 --save ../experiment/X4/EDSR_edsr_s_x4_step2 --pre_train . --pre_train_step1 ../experiment/X4/EDSR_edsr_s_x4_step1/model/model_best.pt --repeat 20 --stage step2 --ext sep
## or train step2 with released checkpoint of step1
python main.py --model edsr_two --scale 4 --batch_size 64 --save ../experiment/X4/EDSR_edsr_s_x4_step2 --pre_train . --pre_train_step1 ../experiment/pre_train/edsr_s_x4_step1.pt --repeat 20 --stage step2 --ext sep