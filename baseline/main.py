# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import datetime
import argparse
import setproctitle
import numpy as np
import cPickle as pickle
import GPUtil
from torch.utils.data import Dataset, DataLoader
import json

from graph import SmapleDataset
from train import run_simple, run_embedding, RnnParameterData, generate_input, generate_input_long_history,get_loc_graph,generate_input_topk,generate_input_history
from train import history_average, most_popular, markov, most_popular_loc,candidate_entropy,candidate_number,static_app,most_recently,most_recently_loc,beyasian_prediction
from model import AppPre,AppPreGtr,AppPreUser,AppPreUserCGtr,AppPreUserPGtr,AppPreUserCGtrTopk,AppPreUserPGtrTopk,AppPreUserRGtr,AppPreUserCGtrHis,AppPreUserCGtrHisAttn,AppPreLocRUserCGtrHis,AppPreLocCUserCGtrHis, AppPreLocRUserCGtrTopk, AppPreLocRUserCGtr
from model import LocPre,LocPreUser,LocPreGtr,LocPreUserGtr,LocPreUserGtrTopk,LocPreUserGtrRec
from model import UserIden
from model import AppPreUserIdenGtr, AppPreLocPreUserIdenGtr, AppPreLocPreGtr, AppPreUserPGtrDocAttn
from model import AppLoss, LocLoss, AppUserLoss, AppLocUserLoss, AppLocLoss
from model import Line_1st, Line_2nd, Line, LocPreUserGtrLocEmb


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(args):
	parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size, tim_emb_size=args.tim_emb_size,
								  app_size=args.app_size, app_encoder_size=args.app_encoder_size,acc_threshold=args.acc_threshold,
								  hidden_size=args.hidden_size, dropout_p=args.dropout_p, top_size=args.top_size,
								  data_name=args.data_name, lr=args.learning_rate,users_start=args.users_start,users_end=args.users_end,
								  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type, loss_alpha=args.loss_alpha,
								  optim=args.optim, lr_schedule=args.lr_schedule, attn_type=args.attn_type, app_emb_mode=args.app_emb_mode,
								  clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode, loss_beta=args.loss_beta,
								  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path, baseline_mode=args.baseline_mode,
								  loc_emb_negative=args.loc_emb_negative, loc_emb_batchsize=args.loc_emb_batchsize, input_mode=args.input_mode, test_start = args.test_start)
	"""metric"""
	T = 10
	T0 = T
	lr = parameters.lr
	metrics = {'train_loss': [], 'valid_loss': [], 'avg_app_auc': [], 'avg_app_map': [], 'avg_app_precision': [], 'avg_app_recall': [], 'avg_loc_top1': [], 'avg_loc_top5': [],'avg_loc_top10': [], 'avg_uid_top1': [], 'avg_uid_top10': [], 'valid_acc': {}}
	if args.candidate == 'loc_entropy':
		candidate = candidate_entropy(parameters.data_neural, start=parameters.users_start, end=parameters.users_end, mode=args.reverse_mode)
	elif args.candidate == 'app_number':
		candidate = candidate_number(parameters.data_neural, start=parameters.users_start, end=parameters.users_end, mode=args.reverse_mode)
	else:
		candidate_tmp = [u for u in parameters.data_neural.keys()]
		candidate = candidate_tmp[parameters.users_start:parameters.users_end]
	parameters.uid_size = max(candidate) + 1
	print('Candidate:',candidate[:5])
	print('Candidate:{} max user_id:{} min user_id:{}'.format(args.candidate, max(candidate), min(candidate)))
	app_number = static_app(parameters,candidate)
	print('App: max app number used:{} min appp number userd:{} average appp number userd:{}'.format(max(app_number), min(app_number), np.mean(app_number)))
	
	"""baseline"""
	if parameters.baseline_mode == 'App':
		print("======Run baseline: beyasian_prediction==========")
		auc, users_auc, f1, users_f1, precision, users_precision, recall, users_recall = beyasian_prediction(parameters,candidate=candidate)
		print('==> NB auc: {:.4f} map: {:.4f} precision: {:.4f} recall: {:.4f}'.format(float(auc), float(f1),float(precision), float(recall)))
		print("======Run baseline: history average==========")
		auc, users_auc, f1, users_f1, precision, users_precision, recall, users_recall = history_average(parameters,candidate=candidate)
		print('==> HA auc: {:.4f} map: {:.4f} precision: {:.4f} recall: {:.4f}'.format(float(auc), float(f1),float(precision), float(recall)))
		print("======Run baseline: most recent==========")
		auc, users_auc, f1, users_f1, precision, users_precision, recall, users_recall = most_recently(parameters,candidate=candidate)
		json.dump(users_f1, open("users_f1_MRU_App.json","w"))	
		print('==> MR auc: {:.4f} map: {:.4f} precision: {:.4f} recall: {:.4f}'.format(float(auc), float(f1),float(precision), float(recall)))
		print("======Run baseline: most popular==========")
		auc, users_auc, f1, users_f1, precision, users_precision, recall, users_recall = most_popular(parameters,candidate=candidate)
		print('==> MF auc: {:.4f} map: {:.4f} precision: {:.4f} recall: {:.4f}'.format(float(auc), float(f1),float(precision), float(recall)))
	elif parameters.baseline_mode == 'Loc':
		print("======Run Loc baseline: one order markov==========")
		avg_acc_top1,avg_acc_top5,avg_acc_top10, user_acc = markov(parameters,candidate=candidate)
		print('==> Markov acc@1: {:.4f} acc@5: {:.4f} acc@10: {:.4f}'.format(float(avg_acc_top1), float(avg_acc_top5),float(avg_acc_top10)))
		print("======Run Loc baseline: most recently==========")
		avg_acc_top1,user_acc  = most_recently_loc(parameters,candidate=candidate)
		json.dump(user_acc, open("avg_acc_top1_MRU_Loc.json","w"))	
		print('==> MR acc@1: {:.4f} '.format(float(avg_acc_top1)))
		print("======Run Loc baseline: most popular==========")
		avg_acc_top1,avg_acc_top5,avg_acc_top10, user_acc  = most_popular_loc(parameters,candidate=candidate)
		print('==> MF acc@1: {:.4f} acc@5: {:.4f} acc@10: {:.4f}'.format(float(avg_acc_top1), float(avg_acc_top5),float(avg_acc_top10)))
	elif parameters.baseline_mode == None:			
		print("================Run models=============")
		"""get loc embedding graph"""
		loc_old2newid = {}
		loc_old2newid,loc_graph = get_loc_graph(parameters, candidate)
		parameters.loc_size = len(loc_old2newid)
		parameters.uid_size = max(candidate)+1
		#Model Training
		print('Split training and testing data', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
		if 'Topk' in parameters.model_mode:
			print('using topk model')
			user_topk = generate_input_topk(parameters,'train',loc_old2newid, mode2=None, candidate=candidate)
		else:
			user_topk = None
		if parameters.input_mode == 'short':
			data_train, train_idx = generate_input(parameters, 'train', loc_old2newid, user_topk, mode2=parameters.history_mode,candidate=candidate)
			data_test, test_idx = generate_input(parameters, 'test', loc_old2newid, user_topk, mode2=parameters.history_mode,candidate=candidate)
		elif parameters.input_mode == 'short_history':
			data_train, train_idx = generate_input_history(parameters, 'train', loc_old2newid, user_topk, mode2=parameters.history_mode,candidate=candidate)
			data_test, test_idx = generate_input_history(parameters, 'test', loc_old2newid, user_topk, mode2=parameters.history_mode,candidate=candidate)
		elif parameters.input_mode == 'long':
			data_train, train_idx = generate_input_long_history(parameters, 'train', loc_old2newid, user_topk, mode2=parameters.history_mode,candidate=candidate)
			data_test, test_idx = generate_input_long_history(parameters, 'test', loc_old2newid, user_topk, mode2=parameters.history_mode,candidate=candidate)
		#print('Generating Line first order similarity')
		#loc_emb_data_1 = SmapleDataset(loc_graph, 0)  #the edge with link, return one pair with label=1
		#loc_emb_data_loaer_1 = DataLoader(loc_emb_data_1, shuffle=True, batch_size=parameters.loc_emb_batchsize, num_workers=4)
		#print('Generating Line second order similarity')
		#loc_emb_data_2 = SmapleDataset(loc_graph, parameters.loc_emb_negative) #negative sample, return 5 pairs with label=-1
		#loc_emb_data_loaer_2 = DataLoader(loc_emb_data_2, shuffle=True, batch_size=parameters.loc_emb_batchsize//6, num_workers=4)
		#print('data len:', len(loc_emb_data_1))
		#T1 = int(len(loc_emb_data_1)//parameters.loc_emb_batchsize)
		
		"""Model Init"""
		print('Model Init!', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
		for mi in range(1):
			#"""Pre App"""
			if parameters.model_mode in ['AppPre']:	
				model = AppPre(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreGtr']:	
				model = AppPreGtr(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreUser']:
				model = AppPreUser(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreUserCGtr']:
				model = AppPreUserCGtr(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreUserRGtr']:
				model = AppPreUserRGtr(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreUserPGtr']:
				model = AppPreUserPGtr(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreUserCGtrTopk']:
				model = AppPreUserCGtrTopk(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreUserCGtrHis']:
				model = AppPreUserCGtrHis(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreUserCGtrHisAttn']:
				model = AppPreUserCGtrHisAttn(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreUserPGtrTopk']:
				model = AppPreUserPGtrTopk(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreLocRUserCGtrHis']:
				model = AppPreLocRUserCGtrHis(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()	
			elif parameters.model_mode in ['AppPreLocCUserCGtrHis']:
				model = AppPreLocCUserCGtrHis(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()	
			elif parameters.model_mode in ['AppPreLocRUserCGtr']:
				model = AppPreLocRUserCGtr(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()	
			elif parameters.model_mode in ['AppPreLocRUserCGtrTopk']:
				model = AppPreLocRUserCGtrTopk(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()	
			elif parameters.model_mode in ['AppPreUserPGtrDocAttn']:
				model = AppPreUserPGtrDocAttn(parameters=parameters).cuda()
				criterion = AppLoss(parameters=parameters).cuda()
			#"""Pre Loc"""	
			elif parameters.model_mode in ['LocPre']:
				model = LocPre(parameters=parameters).cuda()
				criterion = LocLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['LocPreGt','LocPreGtr']:
				model = LocPreGtr(parameters=parameters).cuda()
				criterion = LocLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['LocPreUser']:
				model = LocPreUser(parameters=parameters).cuda()
				criterion = LocLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['LocPreUserGt', 'LocPreUserGtr']: 
				model = LocPreUserGtr(parameters=parameters).cuda()
				criterion = LocLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['LocPreUserGtTopk', 'LocPreUserGtrTopk']:
				model = LocPreUserGtrTopk(parameters=parameters).cuda()
				criterion = LocLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['LocPreUserGtRec', 'LocPreUserGtrRec']:
				model = LocPreUserGtrRec(parameters=parameters).cuda()
				criterion = LocLoss(parameters=parameters).cuda()
			#"""Iden user"""		
			elif parameters.model_mode in ['UserIden']:
				model = UserIden(parameters=parameters).cuda()
				criterion = nn.NLLLoss().cuda()
			#"""Pre App, Loc and user"""		
			elif parameters.model_mode in ['AppPreUserIden']:
				model = AppPreUserIden(parameters=parameters).cuda()
				criterion = AppUserLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreLocPreGtr']:
				model = AppPreLocPreGtr(parameters=parameters).cuda()
				criterion = AppLocLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreUserIdenGt','AppPreUserIdenGtr']:
				model = AppPreUserIdenGtr(parameters=parameters).cuda()
				criterion = AppUserLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreLocPreUserIden']:
				model = AppPreLocPreUserIden(parameters=parameters).cuda()
				criterion = AppLocUserLoss(parameters=parameters).cuda()
			elif parameters.model_mode in ['AppPreLocPreUserIdenGt','AppPreLocPreUserIdenGtr']:
				model = AppPreLocPreUserIdenGtr(parameters=parameters).cuda()
				criterion = AppLocUserLoss(parameters=parameters).cuda()
			#"""For embedding"""		
			elif parameters.model_mode in ['LocEmbed', 'LocPreUserGtrLocEmb']:	
				line_1st = Line_1st(parameters.loc_size, parameters.loc_emb_size).cuda()
				line_2nd = Line_2nd(parameters.loc_size, parameters.loc_emb_size).cuda()
				model = LocPreUserGtrLocEmb(parameters=parameters,line_1st=line_1st,line_2nd=line_2nd,alpha=1).cuda()
				criterion = nn.NLLLoss().cuda()
				T0 = T-1

		print(model)
		params = list(model.parameters())
		k = 0
		for i in params:
			l = 1
			for j in i.size():
				l *= j
			k = k + l
		print("The number of parameters:" + str(k))

		"Forward network with randomly initialization  "
		for epoch in range(1):
			optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=parameters.L2)
			#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=parameters.lr_step, factor=parameters.lr_decay, threshold=1e-3)	
			scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,factor=parameters.lr_decay, threshold=1e-3)
			#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10,20,30], gamma=parameters.lr_decay)
			#prediction_all[epoch] = {}
			#train dataset but test model without feedback
			model, avg_train_loss, prediction = run_simple(data_train, train_idx, 'train_test', parameters.input_mode, lr, parameters.clip, model, optimizer, criterion, parameters.model_mode, parameters.test_start, parameters.acc_threshold)
			print('========================>Epoch:{:0>2d} lr:{}<=================='.format(epoch,lr))
			print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_train_loss, lr))
			metrics['train_loss'].append(avg_train_loss)
			#prediction_all[epoch]['train'] = prediction
			avg_loss, avg_acc, users_acc, prediction = run_simple(data_test, test_idx, 'test', parameters.input_mode, lr, parameters.clip, model,optimizer, criterion, parameters.model_mode, parameters.test_start, parameters.acc_threshold)
			print('==>Test Loss:{:.4f}'.format(avg_loss))
			print('==>Test Acc App_AUC:{:.4f}   App_map:{:.4f}    App_Precision:{:.4f}   App_Recall:{:.4f} '.format(avg_acc['app_auc'], avg_acc['app_map'], avg_acc['app_precision'], avg_acc['app_recall']))
			print('            Loc_top1:{:.4f}  Loc_top5:{:.4f}  Loc_top10:{:.4f}'.format(avg_acc['loc_top1'],avg_acc['loc_top5'], avg_acc['loc_top10']))
			print('            Uid_top1:{:.4f}  Uid_top10:{:.4f}'.format(avg_acc['uid_top1'], avg_acc['uid_top10']))
			metrics['valid_loss'].append(avg_loss) #total average loss
			metrics['valid_acc'][epoch] = users_acc #accuracy for each user
			metrics['avg_app_auc'].append(0) #total average accuracy
			metrics['avg_app_map'].append(0)
			metrics['avg_app_precision'].append(0)
			metrics['avg_app_recall'].append(0)
			metrics['avg_loc_top1'].append(0)
			metrics['avg_loc_top5'].append(0)
			metrics['avg_loc_top10'].append(0)
			metrics['avg_uid_top1'].append(0)
			metrics['avg_uid_top10'].append(0)
			#prediction_all[epoch]['test'] = prediction

		st = time.time()
		start_time = time.time()
		for epoch in range(1, parameters.epoch):
			#prediction_all[epoch] = {}
			if epoch%T < T0:
				#optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=parameters.L2) 		
				#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=parameters.lr_step, factor=parameters.lr_decay, threshold=1e-3)
				model, avg_train_loss, prediction = run_simple(data_train, train_idx, 'train', parameters.input_mode, lr, parameters.clip, model, optimizer, criterion, parameters.model_mode, parameters.test_start, parameters.acc_threshold)
				print('========================>Epoch:{:0>2d} lr:{}<=================='.format(epoch,lr))
				print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_train_loss, lr))
				metrics['train_loss'].append(avg_train_loss)
				#prediction_all[epoch]['train'] = prediction
				avg_loss, avg_acc, users_acc, prediction = run_simple(data_test, test_idx, 'test', parameters.input_mode, lr, parameters.clip, model,optimizer, criterion, parameters.model_mode, parameters.test_start, parameters.acc_threshold)
				print('==>Test Loss:{:.4f}'.format(avg_loss))
				print('==>Test Acc App_AUC:{:.4f}   App_map:{:.4f}    App_Precision:{:.4f}   App_Recall:{:.4f} '.format(avg_acc['app_auc'], avg_acc['app_map'], avg_acc['app_precision'], avg_acc['app_recall']))
				print('            Loc_top1:{:.4f}  Loc_top5:{:.4f}  Loc_top10:{:.4f}'.format(avg_acc['loc_top1'],avg_acc['loc_top5'], avg_acc['loc_top10']))
				print('            Uid_top1:{:.4f}  Uid_top10:{:.4f}'.format(avg_acc['uid_top1'], avg_acc['uid_top10']))
				metrics['valid_loss'].append(avg_loss) #total average loss
				metrics['valid_acc'][epoch] = users_acc #accuracy for each user
				metrics['avg_app_auc'].append(avg_acc['app_auc']) #total average accuracy
				metrics['avg_app_map'].append(avg_acc['app_map'])
				metrics['avg_app_precision'].append(avg_acc['app_precision'])
				metrics['avg_app_recall'].append(avg_acc['app_recall'])
				metrics['avg_loc_top1'].append(avg_acc['loc_top1'])
				metrics['avg_loc_top5'].append(avg_acc['loc_top5'])
				metrics['avg_loc_top10'].append(avg_acc['loc_top10'])
				metrics['avg_uid_top1'].append(avg_acc['uid_top1'])
				metrics['avg_uid_top10'].append(avg_acc['uid_top10'])
				#prediction_all[epoch]['test'] = prediction

				save_name_tmp = 'ep_' + str(epoch) + '_' + str(start_time) + '.m'
				torch.save(model.state_dict(), parameters.save_path + 'tmp/' + save_name_tmp)

				if parameters.lr_schedule == 'Loss':
					if 'AppPre' in parameters.model_mode:
						scheduler.step(avg_acc['app_map'])
					elif 'LocPre' in parameters.model_mode:
						scheduler.step(avg_acc['loc_top1'])
					elif 'UserIden' in parameters.model_mode:
						scheduler.step(avg_acc['uid_top1'])
					lr_last = lr
					lr = optimizer.param_groups[0]['lr']
					if lr_last > lr:
						if 'AppPre' in parameters.model_mode:
							load_epoch = np.argmax(metrics['avg_app_map'])
						elif 'LocPre' in parameters.model_mode:
							load_epoch = np.argmax(metrics['avg_loc_top1'])
						else:
							load_epoch = np.argmax(metrics['avg_uid_top1'])          
						load_name_tmp = 'ep_' + str(load_epoch) + '_' + str(start_time) + '.m'
						model.load_state_dict(torch.load(parameters.save_path + 'tmp/' + load_name_tmp))
						print('load epoch={} model state'.format(load_epoch)) #lr decreased
				if epoch == 1:
					print('single epoch time cost:{}'.format(time.time() - start_time))
				if lr <= 0.9 * 1e-6:
					break

			else:
				optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, line_1st.parameters()), lr=1e-3)
				run_embedding(line_1st, loc_emb_data_loaer_1, 1, optimizer1, 0, epoch, parameters.epoch)
				optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, line_2nd.parameters()), lr=1e-3)
				run_embedding(line_2nd, loc_emb_data_loaer_2, 2, optimizer2, parameters.loc_emb_negative, epoch, parameters.epoch)
				line = Line(line_1st, line_2nd, alpha=1,name='epoch'+str(epoch)) #the ration of 1st and 2nd
				line.save_emb()
		
		overhead = time.time() - start_time
		if 'AppPre' in parameters.model_mode:
			load_epoch = np.argmax(metrics['avg_app_map'])
			print('==>Test Best Epoch:{:0>2d}   App_AUC:{:.4f}   app_map:{:.4f}   App_Precision:{:.4f}   App_Recall:{:.4f} '.format(load_epoch, metrics['avg_app_auc'][load_epoch], metrics['avg_app_map'][load_epoch], metrics['avg_app_precision'][load_epoch], metrics['avg_app_recall'][load_epoch]))
		elif 'LocPre' in parameters.model_mode: 
			load_epoch = np.argmax(metrics['avg_loc_top1'])
			print('==>Test Best Epoch:{:0>2d}   Loc_Top1:{:.4f}   Loc_top10:{:.4f}'.format(load_epoch, metrics['avg_loc_top1'][load_epoch], metrics['avg_loc_top10'][load_epoch]))
		else:
			load_epoch = np.argmax(metrics['avg_uid_top1'])
			print('==>Test Best Epoch:{:0>2d}   Uid_Top1:{:.4f}   Uid_top10:{:.4f}'.format(load_epoch, metrics['avg_uid_top1'][load_epoch], metrics['avg_uid_top10'][load_epoch]))
		load_name_tmp = 'ep_' + str(load_epoch) + '_' + str(start_time) + '.m'
		model.load_state_dict(torch.load(parameters.save_path + 'tmp/' + load_name_tmp))
		save_name = args.model_mode + '_' + str(args.users_start) + '-' + str(args.users_end) + '_' + str(args.uid_emb_size) + '_' + \
					str(args.hidden_size) + '_' + str(args.top_size)+ '_' + \
					str(metrics['avg_app_auc'][load_epoch])[:6] + '_' + str(metrics['avg_app_map'][load_epoch])[:6] + '_' + \
					str(metrics['avg_app_precision'][load_epoch])[:6]+ '_' + str(metrics['avg_app_recall'][load_epoch])[:6] + '_' + \
					str(args.process_name) + '_' + str(args.loss_alpha)[:4] + '_' + str(args.loss_beta)[:4] + '_' + str(overhead/60)[:5]
		#save_name = args.model_mode + '_' + str(args.users_start) + '-' + str(args.users_end) + '_' + str(args.app_encoder_size) + '_' + \
		#			str(args.hidden_size) + '_' + str(metrics['avg_app_map'][load_epoch])[:6] + '_' + \
		#			str(metrics['avg_loc_top1'][load_epoch])[:6]+ '_' + str(metrics['avg_uid_top1'][load_epoch])[:6] + '_' + \
		#			str(args.process_name) + '_' + str(args.loss_alpha)[:4] + '_' + str(args.loss_beta)[:4] + '_' + str(overhead/60)[:5]
		json.dump(metrics['valid_acc'][load_epoch], open("users_acc_"+save_name+".json","w"))	

		"""saving embedding"""
		if parameters.model_mode in ['LocPreUserGtrLocEmb','LocPreUserGtr']:
			model.save_emb()
		
		"""precess visualization"""
		for p in range(1):
			fig = plt.figure(dpi=300)
			if args.plot_mode == 'both':
				ax1 = plt.subplot(221)
				plt.plot(metrics['train_loss'],'r-',label='train_loss')
				plt.plot(metrics['valid_loss'],'b-',label='test_loss')
				plt.legend(loc='best')
				ax2 = plt.subplot(222)
				plt.plot(metrics['avg_app_auc'],'g-',label='test_app_auc')
				plt.plot(metrics['avg_app_map'],'y-',label='test_app_map')
				plt.legend(loc='best')
				ax3 = plt.subplot(223)
				plt.plot(metrics['avg_loc_top1'],'g-',label='test_loc_top1')
				plt.plot(metrics['avg_loc_top10'],'y-',label='test_loc_top10')
				plt.legend(loc='best')
				ax4 = plt.subplot(224)
				plt.plot(metrics['avg_uid_top1'],'g-',label='test_uid_top1')
				plt.plot(metrics['avg_uid_top10'],'y-',label='test_uid_top10')
				plt.legend(loc='best')
			else:
				ax1 = plt.subplot(211)
				plt.plot(metrics['train_loss'],'r-',label='train_loss')
				plt.plot(metrics['valid_loss'],'b-',label='test_loss')
				plt.legend(loc='best')
				ax2 = plt.subplot(212)
				if args.plot_mode == 'App':
					plt.plot(metrics['avg_app_auc'],'g-',label='test_app_auc')
					plt.plot(metrics['avg_app_map'],'y-',label='test_app_map')
				elif args.plot_mode == 'Loc':
					plt.plot(metrics['avg_loc_top1'],'g-',label='test_loc_top1')
					plt.plot(metrics['avg_loc_top10'],'y-',label='test_loc_top10')
				elif args.plot_mode == 'User':
					plt.plot(metrics['avg_uid_top1'],'g-',label='test_uid_top1')
					plt.plot(metrics['avg_uid_top10'],'y-',label='test_uid_top10')
				elif args.plot_mode == 'Loc_emb':
					plt.plot(metrics['avg_loc_emb_P'],'g-',label='avg_loc_emb_P')
					plt.plot(metrics['avg_loc_emb_R'],'y-',label='avg_loc_emb_R')
				plt.legend(loc='best')
			plt.savefig(save_name + '.png')
			precess = np.zeros([10,len(metrics['train_loss'])])
			precess[0,:]=np.array(metrics['train_loss'])
			precess[1,:]=np.array(metrics['valid_loss'])
			precess[2,:]=np.array(metrics['avg_app_auc'])
			precess[3,:]=np.array(metrics['avg_app_map'])
			precess[4,:]=np.array(metrics['avg_loc_top1'])
			precess[5,:]=np.array(metrics['avg_loc_top10'])
			#np.savetxt('result/' + save_name + '_precess_user.txt', precess,fmt='%.4f')
		
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--loc_emb_size', type=int, default=256, help="location embeddings size")
	parser.add_argument('--uid_emb_size', type=int, default=512, help="user id embeddings size")
	parser.add_argument('--tim_emb_size', type=int, default=16, help="time embeddings size")
	parser.add_argument('--app_size', type=int, default=2000, help="original App size")
	parser.add_argument('--app_encoder_size', type=int, default=512, help="App encodering size")
	parser.add_argument('--hidden_size', type=int, default=256)  # should not be too large
	parser.add_argument('--top_size', type=int, default=128)
	parser.add_argument('--attn_size', type=int, default=1024)
	parser.add_argument('--dropout_p', type=float, default=0.3)
	parser.add_argument('--loss_alpha', type=float, default=1) #alpha*target+(1-alpha)*app 
	parser.add_argument('--loss_beta', type=float, default=0) #beta*locloss
	parser.add_argument('--data_name', type=str, default='telecom_4_10_0.8_8_24_10000_exist_Appnumber2000_Locnumber0_compress_30min_tencent_App1to2000_basef0.pk')
	parser.add_argument('--learning_rate', type=float, default=1e-4)  # for attn_user mode:0.0005
	parser.add_argument('--lr_step', type=int, default=2) #patience
	parser.add_argument('--lr_decay', type=float, default=0.1)
	parser.add_argument('--lr_schedule', type=str, default='Loss', choices=['Loss','Keep'])
	parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
	parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
	parser.add_argument('--clip', type=float, default=1.0)
	parser.add_argument('--epoch_max', type=int, default=30)
	parser.add_argument('--history_mode', type=str, default='avg', choices=['max', 'avg', 'whole'])
	parser.add_argument('--attn_type', type=str, default='general', choices=['general', 'concat'])
	parser.add_argument('--data_path', type=str, default='/data/mas/xiatong/preapp/input/')
	parser.add_argument('--save_path', type=str, default='/data/mas/xiatong/preapp/output/')
	parser.add_argument('--candidate', type=str, default=None, choices=['loc_explore', 'app_number', None])
	parser.add_argument('--users_start', type=int, default=0)
	parser.add_argument('--users_end', type=int, default=100) #actuall y user number, index for candidate
	parser.add_argument('--reverse_mode', type=str, default=None, choices=['reverse', None])
	parser.add_argument('--model_mode', type=str, default='AppPreUserPGtrDocAttn',choices=['AppPre','AppPreGtr','AppPreUser','AppPreUserGtr','LocPre','LocPreUser','LocPreGtr','LocPreUserGtr','AppPreLocPre','AppPreLocPreUser','AppPreLocPreGtr','AppPreLocPreUserGtr','LocEmbed','LocPreUserGtrLocEmb','LocPreUserGtrTopk','AppPreUserGtrTopk','AppPreUserGtrRec','LocPreUserGtrRec','AppPreUserCGtr','AppPreUserPGtr','AppPreUserCGtrTopk','AppPreUserPGtrTopk','AppPreUserRGtr','AppPreUserCGtrHis','AppPreUserCGtrHisAttn','AppPreLocRUserCGtrHis','AppPreLocCUserCGtrHis','AppPreUserIdenGtr', 'UserIden','AppPreLocPreUserIdenGtr','AppPreLocRUserCGtrTopk','AppPreLocRUserCGtr','AppPreUserPGtrDocAttn'])
	parser.add_argument('--rnn_type', type=str, default='GRU', choices=['LSTM','RNN','GRU'])
	parser.add_argument('--baseline_mode', type=str, default=None, choices=['App', 'Loc', None])
	parser.add_argument('--plot_mode', type=str, default='App', choices=['App', 'Loc', 'User', 'Loc_emb', 'both'])
	parser.add_argument('--loc_emb_batchsize', type=int, default=1e3)
	parser.add_argument('--loc_emb_negative', type=int, default=5)
	parser.add_argument('--acc_threshold', type=float, default=0.3)
	parser.add_argument('--app_emb_mode', type=str, default='sum',choices=['avg','sum'])
	parser.add_argument('--input_mode', type=str, default='short',choices=['long','short','short_history'])
	parser.add_argument('--test_start', type=int, default=0)
	parser.add_argument('--process_name', type=str, default='preapp_')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	
	a = GPUtil.getAvailable(order='fisrt',limit=8,maxLoad=0.8, maxMemory=0.8, excludeID=[],excludeUUID=[])
	nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	print("Start time:", nowTime)
	flag = 0
	while(len(a)<1):
		if(flag == 0):
			print('non Available GPU!')
		flag += 1
		time.sleep(3)
		a = GPUtil.getAvailable(order='fisrt',limit=8,maxLoad=0.8, maxMemory=0.8, excludeID=[],excludeUUID=[])
		
	print("GPU Available!", nowTime)
	print ('Available GPUs are: ', a)
	os.environ["CUDA_VISIBLE_DEVICES"] = str(a[-1])
	
	setproctitle.setproctitle(args.process_name+'@xiatong')
	
	torch.manual_seed(1)
	np.random.seed(1)
	
	st = time.time()
	run(args)
	print('default setting. Run cost:{}'.format(time.time() - st))
