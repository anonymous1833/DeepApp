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
sparse_traces.py # foursquare data preprocessing
train.py # define tools for train the model
/pretrain
/simple
res.m # pretrained model file
res.rs # detailed evaluation results
res.txt # evaluation results
/simple_long
/attn_local_long
/attn_avg_long_user
/data # preprocessed foursquare sample data (pickle file)
/docs # paper and presentation file
/resutls # the default save path when training the model

##Usage
Load a pretrained model:
python main.py --model_mode=attn_avg_long_user --pretrain=1
The codes contain four network model (simple, simple_long, attn_avg_long_user, attn_local_long) and a baseline model (Markov). The parameter settings for these model can refer to their res.txt file.
