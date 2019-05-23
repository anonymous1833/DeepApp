# DeepApp
This is the code for submission 1833 in CIKM2019

## Datasets
The sample data to evaluate our model can be found in the data folder, which contains 1000+ users and ready for directly used. 

## Requirements
Python 2.7
Pytorch 0.20
cPickle is used in the project to store the preprocessed data and parameters. While appearing some warnings, pytorch 0.3.0 can also be used.

## Project Structure
/codes
main.py
model.py # define models
train.py # define tools for train the model
/data # preprocessed sample data (pickle file)
/baseline #codes for baseline App2Vec


## Usage
python main.py --users_end 1000 --model_mode AppPreLocPreUserIdenGtr --lr_step 1 --process_name user_iden_alpha_beta --hidden_size 512 --app_encoder_size 512 --loss_beta 0.2 --loss_alpha 0.2
The codes contain four network model (DeepApp, DeepApp(App), DeepApp(App+Loc), DeepApp(App+User), RNN) and baseline model (MRU, MFU, HA, Bayes). The parameter settings for these model can refer to run.sh file.
