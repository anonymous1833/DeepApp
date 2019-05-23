# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable
from tqdm import tqdm
import json
import numpy as np
import math
import cPickle as pickle
from collections import deque, Counter
from sklearn import metrics
from graph import Graph, NegativeSampler, SmapleDataset
import time, datetime
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RnnParameterData(object):
	def __init__(self, loc_emb_size=128, uid_emb_size=64, tim_emb_size=16, app_size=20, app_encoder_size=32, hidden_size=128,acc_threshold=0.3,
				 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam', lr_schedule='Loss',
				 history_mode='avg', attn_type='general', model_mode='attn', top_size=16, rnn_type = 'GRU', loss_alpha =0.5,loss_beta =0.01,
				 data_name='', threshold=0.05, epoch_max=30,users_start=0,users_end=100, app_emb_mode ='sum',
				 data_path='', save_path='', baseline_mode=None, loc_emb_negative=5, loc_emb_batchsize=256, input_mode='short',test_start = 3):
		self.data_path = data_path
		self.save_path = save_path
		self.data_name = data_name
		print('===>data load start!',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
		data = pickle.load(open(self.data_path + self.data_name, 'rb'))
		print('===>data load complete!',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
		self.vid_list = data['vid_list']
		self.vid_lookup = data['vid_lookup']
		self.uid_list = data['uid_list']
		self.data_neural = data['data_neural']
		self.users_start = users_start
		self.users_end = users_end

		self.tim_size = 48
		self.loc_size = len(self.vid_list)+1
		self.uid_size = users_end #re_sized for candidate, otherwise, the user embedding will exceed the len(uid)
		self.app_size = app_size
		self.loc_emb_size = loc_emb_size
		self.tim_emb_size = tim_emb_size
		self.uid_emb_size = uid_emb_size
		self.app_encoder_size = app_encoder_size
		self.hidden_size = hidden_size
		self.top_size = top_size
		self.test_start = test_start
		self.acc_threshold = acc_threshold

		self.epoch = epoch_max
		self.dropout_p = dropout_p
		self.use_cuda = True
		self.lr = lr
		self.lr_step = lr_step
		self.lr_decay = lr_decay
		self.lr_schedule = lr_schedule
		self.optim = optim
		self.L2 = L2
		self.clip = clip
		self.loss_alpha = loss_alpha
		self.loss_beta = loss_beta
		self.rnn_type = rnn_type
		self.app_emb_mode = app_emb_mode

		self.model_mode = model_mode
		self.input_mode = input_mode
		self.attn_type = attn_type
		self.history_mode = history_mode
		self.baseline_mode = baseline_mode
		
		self.loc_emb_negative = loc_emb_negative
		self.loc_emb_batchsize = loc_emb_batchsize

"""for user filter"""
def candidate_entropy(data_neural, start=0, end=100, mode=None):
	entropy_users = [(u, data_neural[u]['entropy']) for u in data_neural]
	if mode == 'reverse':
		users_sorted = sorted(entropy_users, key=lambda x: x[1], reverse=True)
	else:
		users_sorted = sorted(entropy_users, key=lambda x: x[1], reverse=False)
	candidate_tmp = users_sorted[int(start):int(end)]
	print('Entropy. Mode:{} max:{} min:{}'.format(mode, candidate_tmp[0], candidate_tmp[-1]))
	candidate = [x[0] for x in candidate_tmp]
	print('candidate user numbers: ', len(candidate))
	return candidate
	
def candidate_number(data_neural, start=0, end=100, mode=None):
	number_users = [(u, data_neural[u]['app_number']) for u in data_neural]
	if mode == 'reverse':
		users_sorted = sorted(number_users, key=lambda x: x[1], reverse=True)
	else:
		users_sorted = sorted(number_users, key=lambda x: x[1], reverse=False)
	candidate_tmp = users_sorted[int(start):int(end)]
	print('App Number. Mode:{} max:{} min:{}'.format(mode, candidate_tmp[0], candidate_tmp[-1]))
	candidate = [x[0] for x in candidate_tmp]
	print('candidate user numbers: ', len(candidate))
	np.savetxt('app_count_user1000.txt', np.array([candidate,[x[1] for x in candidate_tmp]]), fmt='%d')
	return candidate
	
"""for graph"""
def get_loc_graph(parameters, candidate = None):
	loc_old2newid = {}
	loc_newid_lookup = {} #loglat
	lookup_old = parameters.vid_lookup
	reid = 0 
	if candidate == None:
		candidate = parameters.data_neural.keys()
	VisitedGraph = {}
	for u in candidate:
		sessions = parameters.data_neural[u]['sessions']
		for session in sessions: #day
			session = sessions[session]
			for s in session:
				loc = s[1]
				if loc not in loc_old2newid:
					loc_old2newid[loc] = reid
					loc_newid_lookup[reid] = lookup_old[loc]
					reid += 1
				loc_new = loc_old2newid[loc]
				if loc_new not in VisitedGraph:
					VisitedGraph[loc_new] = 1
				else:
					VisitedGraph[loc_new] += 1
	visit_weight = [VisitedGraph[i] for i in range (len(VisitedGraph))]
	VS = sum([visit_weight[n]**(0.75) for n in range (len(VisitedGraph))])
	sample_weight = [int(visit_weight[n]**(0.75)/VS*10e10) for n in range (len(VisitedGraph))]
	#print('sample_weight:',sum(sample_weight))
	#print('number of nodes:',len(loc_newid_lookup.keys()))
	loc_grap = Graph()
	graph = loc_grap.getSpotGraph(loc_newid_lookup)
	#json.dump(loc_newid_lookup, open("loc_newid_lookup.json","w"))	
	return 	loc_old2newid,graph		

"""generate sequence input and target"""
def generate_input(parameters, mode, loc_old2newid, user_topk=None, mode2=None, candidate=None):
	data_train = {}
	train_idx = {}
	data_neural = parameters.data_neural

	if candidate is None:
		candidate = data_neural.keys() #do not filter 
		
	for u in candidate:
		sessions = data_neural[u]['sessions']
		train_id = data_neural[u][mode]  #the day for training
		data_train[u] = {} #re-index user from 0
		locations_train = []
		for c, i in enumerate(train_id): #day
			session = sessions[i]
			trace = {}
			tim_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
			ptim_np = np.reshape(np.array([s[0] for s in session[1:]]), (len(session[:-1]), 1))
			loc_np = np.reshape(np.array([loc_old2newid[s[1]] for s in session[:-1]]), (len(session[:-1]), 1))
			app_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), parameters.app_size))
			loc_target = np.array([loc_old2newid[s[1]] for s in session[1:]])
			loc_loss =  np.array([loc_old2newid[s[1]] for s in session[:-1]])
			app_target = np.reshape(np.array([s[2] for s in session[1:]]), (len(session[:-1]), parameters.app_size))
			uid_target = np.array([u],dtype=np.int).repeat(len(tim_np), 0)
			
			tim_onehot = torch.FloatTensor(len(tim_np),parameters.tim_size)
			tim_onehot.zero_()
			tim_onehot.scatter_(1,torch.LongTensor(tim_np),1)
			
			ptim_onehot = torch.FloatTensor(len(ptim_np),parameters.tim_size)
			ptim_onehot.zero_()
			ptim_onehot.scatter_(1,torch.LongTensor(ptim_np),1)
			
			loc_onehot = torch.FloatTensor(len(loc_np),parameters.loc_size)
			loc_onehot.zero_()
			loc_onehot.scatter_(1,torch.LongTensor(loc_np),1)
			
			trace['loc'] = Variable(torch.LongTensor(loc_np))
			trace['tim'] = Variable(torch.LongTensor(tim_np))
			trace['ptim'] = Variable(torch.LongTensor(ptim_np))
			trace['loc_o'] = Variable(loc_onehot)
			trace['tim_o'] = Variable(tim_onehot)
			trace['ptim_o'] = Variable(ptim_onehot)
			trace['app'] = Variable(torch.FloatTensor(app_np))
			trace['loc_target'] = Variable(torch.LongTensor(loc_target))
			trace['loc_loss'] = Variable(torch.LongTensor(loc_loss))
			trace['app_target'] = Variable(torch.FloatTensor(app_target))
			trace['uid_target'] = Variable(torch.LongTensor(uid_target))
			data_train[u][i] = trace
		train_idx[u] = train_id
		if user_topk is not None:
			data_train[u]['loc_topk'] = Variable(torch.LongTensor(user_topk[u][0]))
			data_train[u]['app_topk'] = Variable(torch.FloatTensor(user_topk[u][1]))
	print(mode,'The number of users: {0} '.format(len(train_idx)))
	return data_train, train_idx #[user ID][train_ID]

def generate_input_topk(parameters,mode,loc_old2newid, mode2=None, candidate=None):
	data_train = {}
	train_idx = {}
	data_neural = parameters.data_neural

	if candidate is None:
		candidate = data_neural.keys() #do not filter 
	
	for u in candidate:
		sessions = data_neural[u]['sessions']
		train_id = data_neural[u][mode]  #the day for training
		locations_train = []
		app_sum = np.zeros((1,parameters.app_size))
		for c, i in enumerate(train_id): #day
			session = sessions[i]
			locations_train.extend([loc_old2newid[s[1]] for s in session])
			app_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), parameters.app_size))
			app_sum = app_sum + np.mean(app_np,0)
		top_app = app_sum/len(train_id)  #do not need to transfer to one-hot
		topk_sort = [x[0] for x in Counter(locations_train).most_common()]
		tmp = topk_sort[-1]
		while len(topk_sort) < 3:
			topk_sort.append(tmp)
		top3_loc = topk_sort[:3]
		if mode == 'train':
			data_train[u] = [top3_loc,top_app]
	return data_train

def generate_input_history(parameters, mode, loc_old2newid, user_topk=None, mode2=None, candidate=None):
	data_train = {}
	train_idx = {}
	data_neural = parameters.data_neural

	if candidate is None:
		candidate = data_neural.keys() #do not filter 
	
	for u in candidate:
		sessions = data_neural[u]['sessions']
		train_id = data_neural[u][mode]
		data_train[u] = {} #re-index user from 0
		for c, i in enumerate(train_id):
			trace = {}
			if mode == 'train' and c == 0:
				continue
			session = sessions[i]
			tim_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
			ptim_np = np.reshape(np.array([s[0] for s in session[1:]]), (len(session[:-1]), 1))
			history_seleted = [s[0] for s in session[1:]]
			loc_np = np.reshape(np.array([loc_old2newid[s[1]] for s in session[:-1]]), (len(session[:-1]), 1))
			app_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), parameters.app_size))
			loc_target = np.array([loc_old2newid[s[1]] for s in session[1:]])
			loc_loss =  np.array([loc_old2newid[s[1]] for s in session[:-1]])
			app_target = np.reshape(np.array([s[2] for s in session[1:]]), (len(session[:-1]), parameters.app_size))
			uid_target = np.array([u],dtype=np.int).repeat(len(tim_np), 0)
			
			history = []
			if mode == 'test':
				test_id = data_neural[u]['train']
				for tt in test_id:
					history.extend([(s[0], s[1], s[2]) for s in sessions[tt]])
			for j in range(c):
				history.extend([(s[0], s[1], s[2]) for s in sessions[train_id[j]]])

			history_count = [0.0]*parameters.tim_size
			history_sum = np.zeros([parameters.tim_size,parameters.app_size])
			history_avg = np.zeros([parameters.tim_size,parameters.app_size])
			for session in history:
				history_sum[session[0],:] += session[2]
				history_count[session[0]] += 1
			for t in range(parameters.tim_size):
				history_avg[t,:] = np.true_divide(history_sum[t,:],history_count[t])
			history_avg[np.isnan(history_avg)] = 0
			
			if 'attn' in parameters.model_mode:
				history_seleted=[]
				for t in range(parameters.tim_size):
					if history_count[t] >0 :
						history_seleted.append(t)
				history_avg=history_avg[history_seleted,:]
			else:
				for t in range(parameters.tim_size):
					if(np.sum(history_avg[t,:])==0):
						history_avg[t,:] = np.mean(history_avg,0) #interpolate with average
				history_avg=history_avg[history_seleted,:]
			history_tim = np.reshape(np.array(history_seleted), (len(history_seleted), 1))
			trace['loc'] = Variable(torch.LongTensor(loc_np))
			trace['tim'] = Variable(torch.LongTensor(tim_np))
			trace['ptim'] = Variable(torch.LongTensor(ptim_np))
			trace['app'] = Variable(torch.FloatTensor(app_np))
			trace['loc_target'] = Variable(torch.LongTensor(loc_target))
			trace['loc_loss'] = Variable(torch.LongTensor(loc_loss))
			trace['app_target'] = Variable(torch.FloatTensor(app_target))
			trace['app_history'] = Variable(torch.FloatTensor(history_avg))
			trace['tim_history'] = Variable(torch.LongTensor(history_tim))
			trace['uid_target'] = Variable(torch.LongTensor(uid_target))
			data_train[u][i] = trace
		train_idx[u] = train_id
		if user_topk is not None:
			data_train[u]['loc_topk'] = Variable(torch.LongTensor(user_topk[u][0]))
			data_train[u]['app_topk'] = Variable(torch.FloatTensor(user_topk[u][1]))
	print(mode,'The number of users: {0}'.format(len(train_idx)))
	return data_train, train_idx

def generate_input_long_history(parameters, mode, loc_old2newid, user_topk=None, mode2=None, candidate=None):
	data_train = {}
	train_idx = {}
	data_neural = parameters.data_neural

	if candidate is None:
		candidate = data_neural.keys() #do not filter 
	
	for u in candidate:
		sessions = data_neural[u]['sessions']
		train_id = data_neural[u][mode]
		data_train[u] = {} #re-index user from 0
		for c, i in enumerate(train_id):
			trace = {}
			if mode == 'train' and c == 0:
				continue
			session = sessions[i]
			loc_target = np.array([loc_old2newid[s[1]] for s in session[1:]])  #the target is only the train_id or test_id day
			#loc_target = np.array([s[1] for s in session[1:]])  #the target is only the train_id or test_id day
			app_target = np.reshape(np.array([s[2] for s in session[1:]]), (len(session[:-1]), parameters.app_size))
			#app_target = app_target*2 -1
			uid_target = np.array([u],dtype=np.int).repeat(len(loc_target), 0)
			loc_loss =  np.array([loc_old2newid[s[1]] for s in session[:-1]])
			
			history = []
			if mode == 'test':
				test_id = data_neural[u]['train']
				for tt in test_id:
					history.extend([(s[0], s[1], s[2]) for s in sessions[tt]])
			for j in range(c):
				history.extend([(s[0], s[1], s[2]) for s in sessions[train_id[j]]])

			history_tim = [t[0] for t in history]
			history_count = [1]
			last_t = history_tim[0]
			count = 1
			for t in history_tim[1:]:
				if t == last_t:
					count += 1
				else:
					history_count[-1] = count
					history_count.append(1)
					last_t = t
					count = 1
			history_tim = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
			history_loc = np.reshape(np.array([loc_old2newid[s[1]] for s in history]), (len(history), 1))
			#history_loc = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
			history_app = np.reshape(np.array([s[2] for s in history]), (len(history), parameters.app_size))
			#history_app = history_app*2 -1
			
			trace['history_loc'] = Variable(torch.LongTensor(history_loc))
			trace['history_tim'] = Variable(torch.LongTensor(history_tim))
			trace['history_app'] = Variable(torch.FloatTensor(history_app))
			trace['history_count'] = history_count

			tim_loc_app = history
			tim_loc_app.extend([(s[0], s[1], s[2]) for s in session]) #extend the current day
			tim_np = np.reshape(np.array([s[0] for s in tim_loc_app[:-1]]), (len(tim_loc_app[:-1]), 1))
			ptim_np = np.reshape(np.array([s[0] for s in tim_loc_app[1:]]), (len(tim_loc_app[:-1]), 1))
			loc_np = np.reshape(np.array([loc_old2newid[s[1]] for s in tim_loc_app[:-1]]), (len(tim_loc_app[:-1]), 1))
			#loc_np = np.reshape(np.array([s[1] for s in tim_loc_app[:-1]]), (len(tim_loc_app[:-1]), 1))
			app_np = np.reshape(np.array([s[2] for s in tim_loc_app[:-1]]), (len(tim_loc_app[:-1]), parameters.app_size))
			#app_np = app_np*2 -1
			trace['loc'] = Variable(torch.LongTensor(loc_np))
			trace['tim'] = Variable(torch.LongTensor(tim_np))
			trace['ptim'] = Variable(torch.LongTensor(ptim_np))
			trace['app'] = Variable(torch.FloatTensor(app_np))
			trace['loc_target'] = Variable(torch.LongTensor(loc_target))
			trace['loc_loss'] = Variable(torch.LongTensor(loc_loss))
			trace['app_target'] = Variable(torch.FloatTensor(app_target))
			trace['uid_target'] = Variable(torch.LongTensor(uid_target))
			data_train[u][i] = trace
		train_idx[u] = train_id
		if user_topk is not None:
			data_train[u]['loc_topk'] = Variable(torch.LongTensor(user_topk[u][0]))
			data_train[u]['app_topk'] = Variable(torch.FloatTensor(user_topk[u][1]))
	print(mode,'The number of users: {0}'.format(len(train_idx)))
	return data_train, train_idx

def generate_queue(train_idx, mode, mode2, inputmode='short'):
	"""return a deque. You must use it by train_queue.popleft()"""
	user = train_idx.keys()
	train_queue = deque()
	if mode == 'random':
		initial_queue = {}
		for u in user:
			if mode2 == 'train':
				if inputmode == 'long' or inputmode == 'short_history':
					initial_queue[u] = deque(train_idx[u][1:]) 
				elif inputmode == 'short':
					initial_queue[u] = deque(train_idx[u])
			else:
				initial_queue[u] = deque(train_idx[u])
		queue_left = 1
		while queue_left > 0:
			np.random.shuffle(user)
			for j, u in enumerate(user):
				if len(initial_queue[u]) > 0:
					train_queue.append((u, initial_queue[u].popleft()))
				if j >= int(0.01 * len(user)):
					break
			queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
	elif mode == 'normal':
		for u in user:
			for i in train_idx[u]:
				train_queue.append((u, i))
	return train_queue

def get_acc2(target, scores): #TOPK acc for loc classification
	"""target and scores are torch cuda Variable"""
	target = target.data.cpu().numpy()
	val, idxx = scores.data.topk(10, 1)
	predx = idxx.cpu().numpy()
	acc = np.zeros(3)
	for i, p in enumerate(predx):
		t = target[i]
		if t in p[:10]:
			acc[2] += 1
		if t in p[:5]:
			acc[1] += 1
		if t == p[0]:
			acc[0] += 1
	return acc

def cal_ap( y_actual, y_pred, k ):
	topK = min( len(y_pred), k )
	# sort by pred
	l_zip = list(zip(y_actual,y_pred))
	s_zip = sorted( l_zip, key=lambda x: x[1], reverse=True )
	# topk of rank result
	s_zip_topk = s_zip[:topK]
	num = 0
	rank = 0
	sumP = 0.0
	for item in s_zip_topk:
		rank += 1
		if item[0] == 1:
			num += 1
			sumP += (num*1.0)/(rank*1.0)
	ap = 0.0
	if num > 0:
		ap = sumP/(num*1.0)
	return ap	

def cal_preb(y_pred, k):
	topK = min( len(y_pred), k )
	# sort by pred
	l_zip = list(zip(range(len(y_pred)),y_pred))
	s_zip = sorted( l_zip, key=lambda x: x[1], reverse=True )
	# topk of rank result
	pre_b = np.zeros(len(y_pred))
	s_zip_topk = s_zip[:topK]
	for (index,sore) in s_zip_topk:
		pre_b[index] = 1
	return pre_b
	
def get_acc(target, scores, threshold): #AUC and F1 for Binary classification
	"""target and scores are torch cuda Variable"""
	target = target.data.cpu().numpy()
	scores = scores.data.cpu().numpy()
	acc = np.zeros(4) #F1-measure, AUC, P, R
	for i in range(len(target)):
		#truth = (target[i,:]+1)/2
		#predict = (scores[i,:]+1)/2
		truth = target[i,:]
		predict = scores[i,:]
		predict_b = cal_preb(predict,5)
		fpr, tpr, thresholds = metrics.roc_curve(truth, predict, pos_label=1)
		acc[0] += metrics.auc(fpr, tpr)
		#acc[1] += metrics.f1_score(truth,predict_b)
		acc[1] += cal_ap(truth,predict,len(truth))
		acc[2] += metrics.precision_score(truth,predict_b)
		acc[3] += metrics.recall_score(truth,predict_b)
	return acc	

"""for training"""
def run_simple(data, run_idx, mode, inputmode, lr, clip, model, optimizer, criterion, mode2=None,test_start=3, threshold=0.3):
	"""mode=train: return model, avg_loss
	   mode=test: return avg_loss,avg_acc,users_rnn_acc"""
	run_queue = None
	if mode == 'train':
		model.train(True)
		run_queue = generate_queue(run_idx, 'random', 'train',inputmode)
		print('Run_queue for training:', len(run_queue)) #(u,i)list
	elif mode == 'test':
		model.train(False)
		run_queue = generate_queue(run_idx, 'normal', 'test',inputmode)
		print('Run_queue for testing:',len(run_queue))
	elif mode == 'train_test': #test the training data set 
		model.train(False)
		run_queue = generate_queue(run_idx, 'random', 'train',inputmode)
		print('Run_queue for training:', len(run_queue)) #(u,i)list
	total_loss = []
	total_loss_app = []
	total_loss_loc = []
	total_loss_uid = []
	queue_len = len(run_queue)
	
	save_prediction = {}

	users_acc = {}
	for c in range(queue_len):
		optimizer.zero_grad()
		u, i = run_queue.popleft() #u is the user, i is the train_ID
		if u not in save_prediction:
			save_prediction[u] = {}
		if u not in users_acc:
			users_acc[u] = {'tim_len':0, 'app_acc':[0,0,0,0], 'loc_acc':[0,0,0], 'uid_acc':[0,0]} 
		loc = data[u][i]['loc'].cuda()
		loc_loss = data[u][i]['loc_loss'].cuda()
		tim = data[u][i]['tim'].cuda()
		ptim = data[u][i]['ptim'].cuda()
		app = data[u][i]['app'].cuda()
		app_loss = data[u][i]['app'].cuda()
		app_target = data[u][i]['app_target'].cuda()
		loc_target = data[u][i]['loc_target'].cuda()
		uid_target = data[u][i]['uid_target'].cuda()
		uid = Variable(torch.LongTensor([u])).cuda()
		#save_prediction[u][i] = {}
		#save_prediction[u][i]['tim'] = tim.data.cpu().numpy()
			
		"""model input"""
		if mode2 in ['AppPreLocPreUserGtrTopk']:	
			app_topk = data[u]['app_topk'].cuda()
			loc_topk = data[u]['loc_topk'].cuda()
			app_scores,loc_scores = model(tim, app, loc, uid, ptim,app_topk,loc_topk)
		elif mode2 in ['AppPreUserGtrTopk','AppPreUserCGtrTopk','AppPreUserPGtrTopk','AppPreUserRGtrTopk','AppPreLocRUserCGtrTopk']:
			app_topk = data[u]['app_topk'].cuda()
			loc_topk = data[u]['loc_topk'].cuda()
			app_scores = model(tim, app, loc, uid, ptim,app_topk,loc_topk)
		elif mode2 in ['AppPreUserCGtrHis','AppPreUserPGtrHis','AppPreUserRGtrHis','AppPreUserCGtrHisAttn','AppPreLocCUserCGtrHis','AppPreLocRUserCGtrHis']:
			app_history = data[u][i]['app_history'].cuda()
			tim_history = data[u][i]['tim_history'].cuda()
			app_scores = model(tim, app, loc, uid, ptim, app_history, tim_history)
		elif mode2 in ['LocPreUserGtrTopk']:
			app_topk = data[u]['app_topk'].cuda()
			loc_topk = data[u]['loc_topk'].cuda()
			loc_scores = model(tim, app, loc, uid, ptim,app_topk,loc_topk)
		elif 'AppPreLocPreUserIdenGtrLinear' in mode2:
			tim_o = data[u][i]['tim_o'].cuda()
			ptim_o = data[u][i]['ptim_o'].cuda()
			loc_o = data[u][i]['loc_o'].cuda()
			app_scores, loc_scores, uid_scores = model(tim_o, app, loc_o, uid, ptim_o)		
		elif 'AppPreLocPreUserIden' in mode2:
			app_scores, loc_scores, uid_scores = model(tim, app, loc, uid, ptim)	
		elif 'AppPreUserIden' in mode2:
			app_scores,uid_scores = model(tim, app, loc, uid, ptim)
		elif 'AppPreLocUserIden' in mode2:
			app_scores,uid_scores = model(tim, app, loc, uid, ptim)
		elif 'AppPreLocPre' in mode2:
			app_scores,loc_scores = model(tim, app, loc, uid, ptim)
		elif 'AppPre' in mode2:
			app_scores = model(tim, app, loc, uid, ptim)
		elif 'LocPre' in mode2:
			loc_scores = model(tim, app, loc, uid, ptim)
		elif 'UserIden' in mode2:
			uid_scores = model(tim, app, loc, uid, ptim)
		elif mode2 in ['LocPreUserGtrLocEmb']:
			loc_scores = model(tim, loc, uid, ptim)
		
		"""RNN output cut"""
		if 'AppPre' in mode2:
			app_scores = app_scores[-(app_target.data.size()[0]-test_start):]
			app_target = app_target[-app_scores.data.size()[0]:]
			app_loss = app_loss[-app_scores.data.size()[0]:]
			targrt_len = len(app_scores)
		if 'LocPre' in mode2:
			loc_scores = loc_scores[-(loc_target.data.size()[0]-test_start):]
			loc_target = loc_target[-loc_scores.data.size()[0]:]
			loc_loss = loc_loss[-loc_scores.data.size()[0]:]
			targrt_len = len(loc_scores)
		if 'UserIden' in mode2:
			uid_scores = uid_scores[-(uid_target.data.size()[0]-test_start):]
			uid_target = uid_target[-uid_scores.data.size()[0]:]
			targrt_len = len(uid_scores)

		"""model loss"""	
		if mode2 in ['AppPre','AppPreGtr','AppPreUser','AppPreUserCGtr','AppPreUserCGtrTopk','AppPreUserPGtr','AppPreUserPGtrTopk','AppPreUserRGtr','AppPreUserRGtr','AppPreUserCGtrHis','AppPreUserCGtrHisAttn','AppPreGtm','AppPreUserGtm','AppPreUserLocGtm','AppPreLocRUserCGtrHis','AppPreLocCUserCGtrHis','AppPreLocRUserCGtrTopk','AppPreLocRUserCGtr']:
			loss = criterion(app_scores, app_target, app_loss)
		elif mode2 in ['LocPre','LocPreGtr','LocPreUser','LocPreUserGtr','LocPreUserGtrLocEmb','LocPreUserGtrRec','LocPreUserGtrTopk']:
			loss = criterion(loc_scores, loc_target, loc_loss)
		elif mode2 in ['UserIden']:
			loss = criterion(uid_scores, uid_target)
			total_loss_uid.append(loss.data.cpu().numpy()[0])
		elif mode2 in ['AppPreUserIdenGtr','AppPreLocUserIdenGtr']:
			loss, loss_app, loss_uid = criterion(app_scores, app_target, uid_scores, uid_target)
			total_loss_app.append(loss_app.data.cpu().numpy()[0])
			total_loss_uid.append(loss_uid.data.cpu().numpy()[0])
		elif mode2 in ['AppPreLocPreGtr']:
			loss, loss_app, loss_loc = criterion(app_scores, app_target, loc_scores, loc_target)
			total_loss_app.append(loss_app.data.cpu().numpy()[0])
			total_loss_loc.append(loss_loc.data.cpu().numpy()[0])
		elif mode2 in ['AppPreLocPreUserIdenGtr','AppPreLocPreUserIdenGtrLinear']:
			loss, loss_app, loss_loc, loss_uid = criterion(app_scores, app_target, loc_scores, loc_target, uid_scores, uid_target)
			total_loss_app.append(loss_app.data.cpu().numpy()[0])
			total_loss_loc.append(loss_loc.data.cpu().numpy()[0])
			total_loss_uid.append(loss_uid.data.cpu().numpy()[0])
			
			
		if mode == 'train':
			loss.backward() 
			try: # gradient clipping
				torch.nn.utils.clip_grad_norm(model.parameters(), clip) #clip = 1, 1 norm
				for p in model.parameters():
					if p.requires_grad:
						p.data.add_(-lr, p.grad.data)
			except:
				pass
			optimizer.step()
		elif mode == 'test':
			users_acc[u]['tim_len'] += targrt_len
			#predict App
			if 'AppPre' in mode2:
				app_acc = get_acc(app_target, app_scores, threshold)
				users_acc[u]['app_acc'][0] += app_acc[0] #AUC
				users_acc[u]['app_acc'][1] += app_acc[1] #F1
				users_acc[u]['app_acc'][2] += app_acc[2] #P
				users_acc[u]['app_acc'][3] += app_acc[3] #R
			if 'LocPre' in mode2:
				loc_acc = get_acc2(loc_target, loc_scores)
				users_acc[u]['loc_acc'][0] += loc_acc[0] #top1
				users_acc[u]['loc_acc'][1] += loc_acc[1] #top5
				users_acc[u]['loc_acc'][2] += loc_acc[2] #top10
			if 'UserIden' in mode2:
				uid_acc = get_acc2(uid_target, uid_scores)
				users_acc[u]['uid_acc'][0] += uid_acc[0] #top1
				users_acc[u]['uid_acc'][1] += uid_acc[2] #top10
		total_loss.append(loss.data.cpu().numpy()[0])

	avg_loss = np.mean(total_loss)
	if len(total_loss_app)>0:
		print('Train App Loss:{:.4f}'.format(np.mean(total_loss_app)))
	if len(total_loss_loc)>0:
		print('Train Loc Loss:{:.4f}'.format(np.mean(total_loss_loc)))
	if len(total_loss_uid)>0:
		print('Train Uid Loss:{:.4f}'.format(np.mean(total_loss_uid)))


	if mode == 'train':
		return model, avg_loss, save_prediction
	elif mode == 'test':
		user_acc = {}  #average acc for each user
		#print('tim_len',[users_acc[u]['tim_len'] for u in users_acc])
		for u in users_acc:
			user_acc[u] = {'app_auc':0, 'app_map':0, 'app_precision':0, 'app_recall':0, 'loc_top1':0, 'loc_top5':0, 'loc_top10':0} 
			user_acc[u]['app_auc'] = users_acc[u]['app_acc'][0] / users_acc[u]['tim_len']
			user_acc[u]['app_map'] = users_acc[u]['app_acc'][1] / users_acc[u]['tim_len']
			user_acc[u]['app_precision'] = users_acc[u]['app_acc'][2] / users_acc[u]['tim_len']
			user_acc[u]['app_recall'] = users_acc[u]['app_acc'][3] / users_acc[u]['tim_len']
			user_acc[u]['loc_top1'] = users_acc[u]['loc_acc'][0] / users_acc[u]['tim_len']
			user_acc[u]['loc_top5'] = users_acc[u]['loc_acc'][1] / users_acc[u]['tim_len']
			user_acc[u]['loc_top10'] = users_acc[u]['loc_acc'][2] / users_acc[u]['tim_len']
			user_acc[u]['uid_top1'] = users_acc[u]['uid_acc'][0] / users_acc[u]['tim_len']
			user_acc[u]['uid_top10'] = users_acc[u]['uid_acc'][1] / users_acc[u]['tim_len']
		avg_acc = {}
		avg_acc['app_auc'] = np.mean([user_acc[u]['app_auc'] for u in users_acc]) 
		avg_acc['app_map'] = np.mean([user_acc[u]['app_map'] for u in users_acc]) 
		avg_acc['app_precision'] = np.mean([user_acc[u]['app_precision'] for u in users_acc]) 
		avg_acc['app_recall'] = np.mean([user_acc[u]['app_recall'] for u in users_acc]) 
		avg_acc['loc_top1'] = np.mean([user_acc[u]['loc_top1'] for u in users_acc])
		avg_acc['loc_top5'] = np.mean([user_acc[u]['loc_top5'] for u in users_acc])
		avg_acc['loc_top10'] = np.mean([user_acc[u]['loc_top10'] for u in users_acc])
		avg_acc['uid_top1'] = np.mean([user_acc[u]['uid_top1'] for u in users_acc])
		avg_acc['uid_top10'] = np.mean([user_acc[u]['uid_top10'] for u in users_acc])
		return avg_loss, avg_acc, user_acc, save_prediction
	elif mode == 'train_test':
		return model, avg_loss, save_prediction

def run_embedding(model, data_loader, order, optimizer, neg_sample_size, epoch, num_epochs):
	data_size = len(data_loader.dataset)
	neg_size = 0 if order == 1 else neg_sample_size

	running_loss = 0.0
	running_num = 0
	with tqdm(total=data_size) as pbar:
		for i, data in enumerate(data_loader):
			x1, x2, w = data
			x1 = x1.view(-1)
			x2 = x2.view(-1)
			w = w.view(-1)
			x1,x2,w = Variable(x1).cuda(),Variable(x2).cuda(),Variable(w).cuda()
			loss = model(x1, x2, w)
			loss *= (neg_size + 1)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.data.cpu().numpy()[0] * x1.size()[0]
			running_num += x1.size()[0]

			update_size = x1.size()[0] // (neg_size + 1)
			pbar.update(update_size)
			pbar.set_description('Epoch %d/%d Train loss: %.4f' % (epoch, num_epochs, running_loss/running_num))

"""for baselines"""
def static_app(parameters,candidate=None):
	data_neural = parameters.data_neural
	if candidate is None:
		candidate = data_neural.keys() #do not filter 
	num= {}
	app_sum = np.zeros(parameters.app_size)
	app_number_session = 0.0
	app_number_session_length = 0.0
	for u in candidate:
		app_list = []
		sessions = data_neural[u]['sessions']
		train_id = data_neural[u]['train']  #the day for training
		for c, i in enumerate(train_id): #day
			session = sessions[i]
			app_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), parameters.app_size))
			app_list.extend(list(np.where(app_np==1)[1]))
			app_sum = app_sum + np.sum(app_np,0)
			app_number_session += len(list(np.where(app_np==1)[1]))
			app_number_session_length += len(app_np)
		num[u] = len(set(app_list))
	app_count = [num[u] for u in candidate]
	print('App: average app number in each session {:.4f}'.format(float(app_number_session/app_number_session_length)))
	print('App: average session number of each user {:.4f}'.format(float(app_number_session_length/len(candidate))))
	#fig = plt.figure(dpi=300)
	#plt.bar(range(parameters.app_size),app_sum)
	#plt.ylim(0,max(app_sum))
	#plt.savefig('app_sum.png')
	np.savetxt('app_count_user.txt', np.array([candidate,app_count]), fmt='%d')
	return app_count

def get_acc_baseline(target, scores, threshold=0.5):
	acc = np.zeros(4) #F1-measure, AUC
	for i in range(len(target)):
		truth = target[i,:]
		predict = scores[i,:]
		#predict_b = np.array([1 if p>=threshold else 0 for p in predict]) #the real data id 0~1
		predict_b = cal_preb(predict,5)
		fpr, tpr, thresholds = metrics.roc_curve(truth, predict, pos_label=1)
		acc[0] += metrics.auc(fpr, tpr)
		#acc[1] += metrics.f1_score(truth,predict_b)
		acc[1] += cal_ap(truth,predict,len(truth))
		acc[2] += metrics.precision_score(truth,predict_b)
		acc[3] += metrics.recall_score(truth,predict_b)
	return acc	

def history_average(parameters,candidate=None):
	time_size = parameters.tim_size #48
	data = parameters.data_neural
	acc = np.zeros((2,1))   # overall acc
	times = 0   # sum up all test day for every tested session
	user_auc = {}
	user_f1 = {}
	user_precision = {}
	user_recall = {}
	
	for user in candidate:
		train_days = len(data[user]['train'])
		test_day = data[user]['test'][0]
		user_train_data = np.zeros((train_days, time_size, parameters.app_size))    # data for every user in every day and evry time slice
		user_test_data = np.zeros((time_size, parameters.app_size))     # user test data

		train_day_i = 0
		for train_day in data[user]['train']:
			for session in data[user]['sessions'][train_day]:
				user_train_data[train_day_i, session[0], :] = session[2]
			train_day_i += 1

		user_train_data = np.sum(user_train_data, axis=0)   # sum up train days' data into one day
		scores = np.true_divide(user_train_data, train_days)
		scores[np.isnan(scores)] = 0
		
		test_session = []
		for session in data[user]['sessions'][test_day]:
			user_test_data[session[0],:] = session[2]
			test_session.append(session[0])
		test_days = np.sum(np.any(user_test_data, axis=1)) #The number of test sessions
		user_acc = get_acc_baseline(user_test_data[test_session,:], scores[test_session,:], parameters.acc_threshold)
		user_auc[user] = np.true_divide(user_acc[0], test_days) 
		user_f1[user] = np.true_divide(user_acc[1], test_days) 
		user_precision[user] = np.true_divide(user_acc[2], test_days) 
		user_recall[user] = np.true_divide(user_acc[3], test_days) 
	ave_auc = np.mean([user_auc[x] for x in user_auc])
	ave_f1 = np.mean([user_f1[x] for x in user_f1])
	ave_precision = np.mean([user_precision[x] for x in user_precision])
	ave_recall = np.mean([user_recall[x] for x in user_recall])
	return ave_auc,user_auc,ave_f1,user_f1,ave_precision, user_precision, ave_recall, user_recall

def most_popular(parameters,candidate=None):
	time_size = parameters.tim_size #48
	data = parameters.data_neural
	acc = np.zeros((2,1))   # overall acc
	times = 0   # sum up all test day for every tested session
	user_auc = {}
	user_f1 = {}
	user_precision = {}
	user_recall = {}
	
	for user in candidate:
		train_days = len(data[user]['train'])
		test_day = data[user]['test'][0]
		user_train_data = np.zeros((train_days, time_size, parameters.app_size))    # data for every user in every day and every time slice
		time_slice_days = np.zeros((time_size, 1))    # sum of days in which specific time slice exist
		user_test_data = np.zeros((time_size, parameters.app_size))     # user test data
		#user_acc = np.zeros((2,1))      # acc for each user

		train_day_i = 0
		for train_day in data[user]['train']:
			for session in data[user]['sessions'][train_day]:
				user_train_data[train_day_i, session[0], :] = session[2]
			train_day_i += 1

		scores = np.sum(user_train_data, axis=(0, 1))  #add all train sessions
		scores = [0 if x < scores[np.argsort(-scores)[4]] else 1 for x in scores]
		#print('Top 5 App for each user:', user,sum(scores))
		scores = np.tile(scores, (time_size, 1))

		test_session = []
		for session in data[user]['sessions'][test_day]:
			user_test_data[session[0],:] = session[2]
			test_session.append(session[0])
		test_days = np.sum(np.any(user_test_data, axis=1))
		
		user_acc = get_acc_baseline(user_test_data[test_session,:], scores[test_session,:], parameters.acc_threshold)
		user_auc[user] = np.true_divide(user_acc[0], test_days) 
		user_f1[user] = np.true_divide(user_acc[1], test_days) 
		user_precision[user] = np.true_divide(user_acc[2], test_days) 
		user_recall[user] = np.true_divide(user_acc[3], test_days) 
	ave_auc = np.mean([user_auc[x] for x in user_auc])
	ave_f1 = np.mean([user_f1[x] for x in user_f1])
	ave_precision = np.mean([user_precision[x] for x in user_precision])
	ave_recall = np.mean([user_recall[x] for x in user_recall])
	return ave_auc,user_auc,ave_f1,user_f1,ave_precision, user_precision, ave_recall, user_recall

def most_recently(parameters,candidate=None):
	data_neural = parameters.data_neural
	user_auc = {}
	user_f1 = {}
	user_precision = {}
	user_recall = {}
	for u in candidate:
		sessions = data_neural[u]['sessions']
		test_id = data_neural[u]['test']  #the day for training
		for c, i in enumerate(test_id): #day
			session = sessions[i]
			app_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), parameters.app_size))
			tim_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
			ptim_np = np.reshape(np.array([s[0] for s in session[1:]]), (len(session[:-1]), 1))
			app_target = np.reshape(np.array([s[2] for s in session[1:]]), (len(session[:-1]), parameters.app_size))
		app_score = np.zeros([len(app_target),parameters.app_size])
		for t in range(len(tim_np)):
			if(ptim_np[t] == (tim_np[t]+1)):
				app_score[t] = app_np[t]
		user_acc = get_acc_baseline(app_target,app_score, parameters.acc_threshold)
		user_auc[u] = user_acc[0]/len(app_target) 
		user_f1[u] = user_acc[1]/len(app_target)
		user_precision[u] = user_acc[2]/len(app_target) 
		user_recall[u] = user_acc[3]/len(app_target)
	ave_auc = np.mean([user_auc[x] for x in user_auc])
	ave_f1 = np.mean([user_f1[x] for x in user_f1])
	ave_precision = np.mean([user_precision[x] for x in user_precision])
	ave_recall = np.mean([user_recall[x] for x in user_recall])
	return ave_auc,user_auc,ave_f1,user_f1,ave_precision, user_precision, ave_recall, user_recall

def beyasian_prediction(parameters,candidate=None):
	time_size = parameters.tim_size #48
	data = parameters.data_neural
	times = 0   # sum up all test day for every tested session
	user_auc = {}
	user_f1 = {}
	user_precision = {}
	user_recall = {}
	
	for user in candidate:
		Tim_NB = {} #beyasian probability
		Loc_NB = {}
		App_NB = np.zeros([parameters.app_size,parameters.app_size])

		for train_day in data[user]['train']:
			session = data[user]['sessions'][train_day]
			tim_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
			ptim_np = np.reshape(np.array([s[0] for s in session[1:]]), (len(session[:-1]), 1))
			loc_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
			app_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), parameters.app_size))
			app_target = np.reshape(np.array([s[2] for s in session[1:]]), (len(session[:-1]), parameters.app_size))
			for i in range(len(tim_np)):
				ptim,loc,app,papp = ptim_np[i,0],loc_np[i,0],app_np[i,:],app_target[i,:]
				if ptim not in Tim_NB:
					Tim_NB[ptim] = [0]*parameters.app_size
				if loc not in Loc_NB:
					Loc_NB[loc] = [0]*parameters.app_size
				Tim_NB[ptim] += papp
				Loc_NB[loc] += papp
				last_app = np.where(app==1)[0].tolist() #the last apps
				next_app = np.where(papp==1)[0].tolist() #the next apps
				continue_app = set(last_app)&set(next_app) #still used apps
				giveup_app = set(last_app) - continue_app #not used apps
				new_app = set(next_app) - continue_app #new open apps
				for a in continue_app:
					App_NB[a,a] += 1
				for a in giveup_app:
					for b in new_app:
						#App_NB[a,b] += 1.0/len(new_app)
						App_NB[a,b] += 1
						
			#normalization
			for loc in Loc_NB:
				Loc_NB[loc] = Loc_NB[loc]*1.0/sum(Loc_NB[loc])
			for tim in Tim_NB:
				Tim_NB[tim] = Tim_NB[tim]*1.0/sum(Tim_NB[tim])
			for i in range(parameters.app_size):
				if(sum(App_NB[i,:])>0):
					App_NB[i,:] = App_NB[i,:]/sum(App_NB[i,:])
		#predict
		test_session = 0
		target_app = []
		score_app = []
		score_tim = []
		score_loc = []
		scores = []
		for test_day in data[user]['test']:
			session = data[user]['sessions'][test_day]
			tim_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
			ptim_np = np.reshape(np.array([s[0] for s in session[1:]]), (len(session[:-1]), 1))
			loc_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
			app_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), parameters.app_size))
			app_target = np.reshape(np.array([s[2] for s in session[1:]]), (len(session[:-1]), parameters.app_size))
			for i in range(len(tim_np)):
				ptim,loc, app, papp = ptim_np[i,0],loc_np[i,0],app_np[i,:],app_target[i,:]
				target_app.append(papp)
				# if not appear in training data, random select one
				if ptim not in Tim_NB:      
					ptim = Tim_NB.keys()[np.random.randint(0,len(Tim_NB))]
				if loc not in Loc_NB:
					loc = Loc_NB.keys()[np.random.randint(0,len(Loc_NB))]
				score_tim.append(Tim_NB[ptim])
				score_loc.append(Tim_NB[ptim]*Loc_NB[loc])
				score_app_temp = [0]*parameters.app_size
				last_app = np.where(app==1)[0].tolist() #the last apps
				for i in range(parameters.app_size):
					for a in last_app:
						score_app_temp[i] += App_NB[a,i]
				score_app.append(Tim_NB[ptim]*score_app_temp)
				scores.append(Tim_NB[ptim]*Loc_NB[loc]*score_app_temp) #utilize all features
			test_session += len(tim_np)
		user_acc_tim = get_acc_baseline(np.array(target_app), np.array(score_tim), parameters.acc_threshold)
		user_acc_loc = get_acc_baseline(np.array(target_app), np.array(score_loc), parameters.acc_threshold)
		user_acc_app = get_acc_baseline(np.array(target_app), np.array(score_app), parameters.acc_threshold)
		user_acc_all = get_acc_baseline(np.array(target_app), np.array(scores), parameters.acc_threshold)
		user_auc[user] = [user_acc_tim[0]/test_session,user_acc_loc[0]/test_session,user_acc_app[0]/test_session,user_acc_all[0]/test_session] 
		user_f1[user] = [user_acc_tim[1]/test_session,user_acc_loc[1]/test_session,user_acc_app[1]/test_session,user_acc_all[1]/test_session] 
		user_precision[user] = [user_acc_tim[2]/test_session,user_acc_loc[2]/test_session,user_acc_app[2]/test_session,user_acc_all[2]/test_session] 
		user_recall[user] = [user_acc_tim[3]/test_session,user_acc_loc[3]/test_session,user_acc_app[3]/test_session,user_acc_all[3]/test_session] 
		
	auc = np.mean([user_auc[x][0] for x in user_auc])
	f1 = np.mean([user_f1[x][0] for x in user_f1])
	precision = np.mean([user_precision[x][0] for x in user_precision])
	recall = np.mean([user_recall[x][0] for x in user_recall])	
	print('====> Tim Beyasian  auc: {:.4f} map: {:.4f} precision: {:.4f} recall: {:.4f}'.format(float(auc), float(f1),float(precision), float(recall)))
	
	auc = np.mean([user_auc[x][1] for x in user_auc])
	f1 = np.mean([user_f1[x][1] for x in user_f1])
	precision = np.mean([user_precision[x][1] for x in user_precision])
	recall = np.mean([user_recall[x][1] for x in user_recall])	
	print('====> Loc Beyasian  auc: {:.4f} map: {:.4f} precision: {:.4f} recall: {:.4f}'.format(float(auc), float(f1),float(precision), float(recall)))
	
	auc = np.mean([user_auc[x][2] for x in user_auc])
	f1 = np.mean([user_f1[x][2] for x in user_f1])
	precision = np.mean([user_precision[x][2] for x in user_precision])
	recall = np.mean([user_recall[x][2] for x in user_recall])	
	print('====> App Beyasian  auc: {:.4f} map: {:.4f} precision: {:.4f} recall: {:.4f}'.format(float(auc), float(f1),float(precision), float(recall)))
		
	ave_auc = np.mean([user_auc[x][3] for x in user_auc])
	ave_f1 = np.mean([user_f1[x][3] for x in user_f1])
	ave_precision = np.mean([user_precision[x][3] for x in user_precision])
	ave_recall = np.mean([user_recall[x][3] for x in user_recall])
	return ave_auc,user_auc,ave_f1,user_f1,ave_precision, user_precision, ave_recall, user_recall
	
def markov(parameters,candidate=None):
	data = parameters.data_neural
	test_start = parameters.test_start
	validation = {}
	for u in candidate:
		traces = parameters.data_neural[u]['sessions']
		train_id = parameters.data_neural[u]['train']
		test_id = parameters.data_neural[u]['test']
		trace_train = []
		for tr in train_id:
			trace_train.append([t[1] for t in traces[tr]])
		locations_train = []
		for t in trace_train:
			locations_train.extend(t)
		trace_test = []
		for tr in test_id:
			trace_test.append([t[1] for t in traces[tr]])
		locations_test = []
		for t in trace_test:
			locations_test.extend(t)
		validation[u] = [locations_train, locations_test]
	acc_top1 = 0
	acc_top10 = 0
	count = 0
	user_acc = {}
	for u in validation.keys():
		topk = list(set(validation[u][0])) #histroy visited locations
		transfer = np.zeros((len(topk), len(topk)))

		# train
		sessions = parameters.data_neural[u]['sessions']
		train_id = parameters.data_neural[u]['train']
		for i in train_id:
			for j, s in enumerate(sessions[i][:-1]):
				loc = s[1]
				target = sessions[i][j + 1][1]
				if loc in topk and target in topk:
					r = topk.index(loc)
					c = topk.index(target)
					transfer[r, c] += 1
		for i in range(len(topk)):
			tmp_sum = np.sum(transfer[i, :])
			if tmp_sum > 0:
				transfer[i, :] = transfer[i, :] / tmp_sum
		#print(u,transfer)
		# validation
		user_count = 0
		user_acc[u] = [0,0,0,0] #top1, top5, top10
		test_id = parameters.data_neural[u]['test']
		for i in test_id:
			for j, s in enumerate(sessions[i][:-1]):
				loc = s[1]  #the last location
				target = sessions[i][j + 1][1] #the next location
				if j >= test_start:
					user_count += 1
					if loc in topk: 
						pred = np.argsort(transfer[topk.index(loc), :]) #top1
						pred2 = topk[pred[-1]] #top1
						if pred2 == target:
							user_acc[u][0] += 1
						pred3 = np.array(topk)[pred[-5:]] #top5
						if target in pred3:
							user_acc[u][1] += 1
						pred4 = np.array(topk)[pred[-10:]] #top10
						if target in pred4:
							user_acc[u][2] += 1
		user_acc[u][0] = user_acc[u][0] / user_count
		user_acc[u][1] = user_acc[u][1] / user_count
		user_acc[u][2] = user_acc[u][2] / user_count
		user_acc[u][3] = len(transfer)
	avg_acc_top1 = np.mean([user_acc[u][0] for u in user_acc])
	avg_acc_top5 = np.mean([user_acc[u][1] for u in user_acc])
	avg_acc_top10 = np.mean([user_acc[u][2] for u in user_acc])
	avg_locnum_user = np.mean([user_acc[u][3] for u in user_acc])
	print('****Lob/User=:',avg_locnum_user)
	return avg_acc_top1,avg_acc_top5, avg_acc_top10, user_acc

def most_popular_loc(parameters,candidate=None):
	data = parameters.data_neural
	validation = {}
	test_start = parameters.test_start
	for u in candidate:
		traces = parameters.data_neural[u]['sessions']
		train_id = parameters.data_neural[u]['train']
		test_id = parameters.data_neural[u]['test']
		trace_train = []
		for tr in train_id:
			trace_train.append([t[1] for t in traces[tr]])
		locations_train = []
		for t in trace_train:
			locations_train.extend(t)
		trace_test = []
		for tr in test_id:
			trace_test.append([t[1] for t in traces[tr]])
		locations_test = []
		for t in trace_test:
			locations_test.extend(t)
		validation[u] = [locations_train, locations_test]
	acc_top1 = 0
	acc_top10 = 0
	count = 0
	user_acc = {}
	for u in validation.keys():
		# train
		sessions = parameters.data_neural[u]['sessions']
		train_id = parameters.data_neural[u]['train']
		topk = list(set(validation[u][0]))
		topk_sort = Counter(validation[u][0]).most_common()
		top5 = [x[0] for x in topk_sort[:5]]
		top10 = [x[0] for x in topk_sort[:10]]
		# validation
		user_count = 0
		user_acc[u] = [0,0,0] #top1, top10
		test_id = parameters.data_neural[u]['test']
		for i in test_id:
			for j, s in enumerate(sessions[i][:-1]):
				loc = s[1]  #the last location
				target = sessions[i][j + 1][1] #the next location
				if j >= test_start:
					count += 1
					user_count += 1
					if loc in topk: # 
						pred2 = top10[0]
						if pred2 == target:
							user_acc[u][0] += 1
						pred3 = top5 #top5
						if target in pred3:
							user_acc[u][1] += 1
						pred4 = top10 #top10
						if target in pred4:
							user_acc[u][2] += 1
		user_acc[u][0] = user_acc[u][0] / user_count
		user_acc[u][1] = user_acc[u][1] / user_count
		user_acc[u][2] = user_acc[u][2] / user_count
	avg_acc_top1 = np.mean([user_acc[u][0] for u in user_acc])
	avg_acc_top5 = np.mean([user_acc[u][1] for u in user_acc])
	avg_acc_top10 = np.mean([user_acc[u][2] for u in user_acc])
	return avg_acc_top1,avg_acc_top5,avg_acc_top10, user_acc
	
def most_recently_loc(parameters,candidate=None):
	user_acc = {}
	for u in candidate:
		user_acc[u] = 0
		sessions = parameters.data_neural[u]['sessions']
		test_id = parameters.data_neural[u]['test']  #the day for training
		for c, i in enumerate(test_id): #day
			session = sessions[i]
			loc_np = np.array([s[1] for s in session[:-1]])
			loc_target = np.array([s[1] for s in session[1:]])
			for i in range (len(loc_np)):
				if loc_np[i] == loc_target[i]:
					user_acc[u] += 1
		user_acc[u] = user_acc[u]/ len(loc_np)
	avg_acc_top1 = np.mean([user_acc[u] for u in user_acc])
	return avg_acc_top1, user_acc	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	