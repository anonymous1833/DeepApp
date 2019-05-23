#!/bin/bash 
#for b in 0.1 0.2 0.5 1
#do
#for a in 0 0.05 0.1 0.2 0.5 1
#python main.py --users_end 1000 --model_mode AppPreLocPreUserIdenGtr --lr_step 1 --process_name user_iden_alpha_beta --hidden_size 512 --app_encoder_size 512 --loss_beta $a --loss_alpha $b
#done 
#done 

python main.py --users_end 1000 --model_mode AppPreLocPreUserIdenGtr --lr_step 1 --process_name user_iden_alpha_beta --hidden_size 512 --app_encoder_size 512 --loss_beta 1.0 --loss_alpha 0.2
python main.py --users_end 1000 --model_mode AppPreLocPreUserIdenGtr --lr_step 1 --process_name user_iden_alpha_beta --hidden_size 512 --app_encoder_size 512 --loss_beta 0.1 --loss_alpha 0
python main.py --users_end 1000 --model_mode AppPreLocPreUserIdenGtr --lr_step 1 --process_name user_iden_alpha_beta --hidden_size 512 --app_encoder_size 512 --loss_beta 0.2 --loss_alpha 0
python main.py --users_end 1000 --model_mode AppPreLocPreUserIdenGtr --lr_step 1 --process_name user_iden_alpha_beta --hidden_size 512 --app_encoder_size 512 --loss_beta 0.5 --loss_alpha 0
python main.py --users_end 1000 --model_mode AppPreLocPreUserIdenGtr --lr_step 1 --process_name user_iden_alpha_beta --hidden_size 512 --app_encoder_size 512 --loss_beta 0.1 --loss_alpha 0

 python main.py --data_name telecom_4_10_0.8_8_24_1000_exist_Appnumber2000_Locnumber0_compress_30min_tencent_App1to2000_basef0.pk --model_mode AppPreLocPreUserIdenGtrLinear --users_end 1000 --uid_emb_size 256 --hidden_size 512 --top_size 128 --loss_alpha 0.2 --loss_beta 0.2

