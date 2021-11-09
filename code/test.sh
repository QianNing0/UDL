# !/bin/sh
## step1
python main.py --test_only --data_test Set5 --testset Set5 --scale 4 --model edsr --pre_train ../experiment/pre_train/edsr_s_x4_step1.pt --save ../experiment/X4/edsr_s_x4_step1_test
python main.py --test_only --data_test Set14 --testset Set14 --scale 4 --model edsr --pre_train ../experiment/pre_train/edsr_s_x4_step1.pt --save ../experiment/X4/edsr_s_x4_step1_test
python main.py --test_only --data_test BSD100 --testset BSD100 --scale 4 --model edsr --pre_train ../experiment/pre_train/edsr_s_x4_step1.pt --save ../experiment/X4/edsr_s_x4_step1_test
python main.py --test_only --data_test Urban100 --testset Urban100 --scale 4 --model edsr --pre_train ../experiment/pre_train/edsr_s_x4_step1.pt --save ../experiment/X4/edsr_s_x4_step1_test
python main.py --test_only --data_test Manga109 --testset Manga109 --scale 4 --model edsr --pre_train ../experiment/pre_train/edsr_s_x4_step1.pt --save ../experiment/X4/edsr_s_x4_step1_test

## step2
python main.py --test_only --data_test Set5 --testset Set5 --scale 4 --model edsr_two --pre_train ../experiment/pre_train/edsr_s_x4_step2.pt --save ../experiment/X4/edsr_s_x4_step2_test
python main.py --test_only --data_test Set14 --testset Set14 --scale 4 --model edsr_two --pre_train ../experiment/pre_train/edsr_s_x4_step2.pt --save ../experiment/X4/edsr_s_x4_step2_test
python main.py --test_only --data_test BSD100 --testset BSD100 --scale 4 --model edsr_two --pre_train ../experiment/pre_train/edsr_s_x4_step2.pt --save ../experiment/X4/edsr_s_x4_step2_test
python main.py --test_only --data_test Urban100 --testset Urban100 --scale 4 --model edsr_two --pre_train ../experiment/pre_train/edsr_s_x4_step2.pt --save ../experiment/X4/edsr_s_x4_step2_test
python main.py --test_only --data_test Manga109 --testset Manga109 --scale 4 --model edsr_two --pre_train ../experiment/pre_train/edsr_s_x4_step2.pt --save ../experiment/X4/edsr_s_x4_step2_test
