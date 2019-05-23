# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# ############# simple dnn model ####################### #
class AppPreGtm(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, parameters):
		super(AppPreGtm, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.tim_emb_size + self.uid_emb_size
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(input_size, self.app_size)

	def forward(self, tim, app, loc, uid, ptim):
		ptim_emb = self.emb_tim(ptim).squeeze(1)
		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		x = torch.cat((ptim_emb,uid_emb), 1)
		x = self.dropout(x)
		out = self.dec_app(x)
		score = F.sigmoid(out)

		return score

class AppPreUserGtm(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, parameters):
		super(AppPreUserGtm, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.uid_emb_size
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(input_size, self.app_size)

	def forward(self, tim, app, loc, uid, ptim):
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim).squeeze(1)
		ptim_emb = self.emb_tim(ptim).squeeze(1)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb
		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		x = torch.cat((tim_emb,app_emb,ptim_emb,uid_emb), 1)
		x = self.dropout(x)
		out = self.dec_app(x)
		score = F.sigmoid(out)

		return score

class AppPreUserLocGtm(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, parameters):
		super(AppPreUserLocGtm, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.uid_emb_size + self.loc_emb_size
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(input_size, self.app_size)

	def forward(self, tim, app, loc, uid, ptim):
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim).squeeze(1)
		ptim_emb = self.emb_tim(ptim).squeeze(1)
		loc_emb = self.emb_loc(loc).squeeze(1)
		
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb
		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		x = torch.cat((tim_emb,app_emb,ptim_emb,uid_emb,loc_emb), 1)
		x = self.dropout(x)
		out = self.dec_app(x)
		score = F.sigmoid(out)

		return score

# ############# simple rnn model ####################### #
class AppPre(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, parameters):
		super(AppPre, self).__init__()
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = self.dropout(app_emb)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		out = self.dec_app(out)
		#score = F.selu(out)
		#score = F.tanh(out)
		score = F.sigmoid(out)

		return score

class AppPreUser(nn.Module):
	"""baseline rnn model, only use time, APP usage, with user embedding"""

	def __init__(self, parameters):
		super(AppPreUser, self).__init__()
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size + self.uid_emb_size, self.app_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = self.dropout(app_emb)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		out = torch.cat((out, uid_emb), 1)
		out = self.dec_app(out)
		#score = F.selu(out)
		#score = F.tanh(out)
		score = F.sigmoid(out)

		return score

class AppPreGtr(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, parameters):
		super(AppPreGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		#for i in range(len(app)):
		#	app_emb[i,:] = app[i,:].view(1,self.app_size).mm(self.emb_app.weight)/sum(app[i,:])
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb,ptim_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		out = self.dec_app(out)
		#score = F.selu(out)
		#score = F.tanh(out)
		score = F.sigmoid(out)

		return score

class AppPreUserCGtr(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, parameters):
		super(AppPreUserCGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size + self.uid_emb_size, self.app_size)
		#self.dec_app = nn.Linear(self.hidden_size, self.app_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb,ptim_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		out = torch.cat((out, uid_emb), 1)
		#out = torch.mul(out, uid_emb)
		out = self.dec_app(out)
		score = F.sigmoid(out)

		return score

	def save_emb(self):
		np.save('AppPreUserGtr.npy',self.emb_app.weight.data.cpu().numpy())
		print('********save************')

class AppPreUserPGtr(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, parameters):
		super(AppPreUserPGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.hidden_size = parameters.hidden_size
		assert self.hidden_size==self.uid_emb_size,'hidden_size!=uid_emb_size,cannot product'
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		#self.dec_app = nn.Linear(self.hidden_size + self.uid_emb_size, self.app_size)
		self.dec_app = nn.Linear(self.hidden_size, int(self.app_size/2))
		self.dec_app2 = nn.Linear(int(self.app_size/2), self.app_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb,ptim_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		#out = torch.cat((out, uid_emb), 1)
		out = torch.mul(out, uid_emb)
		out = self.dec_app(out)
		out = self.dec_app2(out)
		score = F.sigmoid(out)

		return score

	def save_emb(self):
		np.save('AppPreUserGtr.npy',self.emb_app.weight.data.cpu().numpy())
		print('********save************')

class AppPreUserRGtr(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, parameters):
		super(AppPreUserRGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.uid_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1).unsqueeze(1)
		x = torch.cat((tim_emb, app_emb,ptim_emb,uid_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		#uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		#out = torch.cat((out, uid_emb), 1)
		#out = torch.mul(out, uid_emb)
		out = self.dec_app(out)
		#out = self.dec_app2(out)
		score = F.sigmoid(out)

		return score

	def save_emb(self):
		np.save('AppPreUserGtr.npy',self.emb_app.weight.data.cpu().numpy())
		print('********save************')

class AppPreUserCGtrTopk(nn.Module):
	"""baseline rnn model, only use time, APP usage, with user embedding"""

	def __init__(self, parameters):
		super(AppPreUserCGtrTopk, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.top_size = parameters.top_size
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.fc_top = nn.Linear(self.app_emb_size, self.top_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size + self.uid_emb_size + self.top_size, self.app_size)
		
	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim, app_topk, loc_topk):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb, ptim_emb), 2)
		x = self.dropout(x)
		
		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		top_emb = app_topk.mm(self.emb_app.weight)
		app_top = self.fc_top(top_emb).repeat(tim.size()[0], 1)
		out = torch.cat((out, uid_emb, app_top), 1)

		out = self.dec_app(out)
		#score = F.selu(out)
		#score = F.tanh(out)
		score = F.sigmoid(out)

		return score

class AppPreUserPGtrTopk(nn.Module):
	"""baseline rnn model, only use time, APP usage, with user embedding"""

	def __init__(self, parameters):
		super(AppPreUserPGtrTopk, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.top_size = parameters.top_size
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.fc_top = nn.Linear(self.app_emb_size, self.top_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size + self.top_size, int(self.app_size/2))
		self.dec_app2 = nn.Linear(int(self.app_size/2), self.app_size)
		
	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim, app_topk, loc_topk):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb, ptim_emb), 2)
		x = self.dropout(x)
		
		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		top_emb = app_topk.mm(self.emb_app.weight)
		app_top = self.fc_top(top_emb).repeat(tim.size()[0], 1)
		
		out = torch.mul(out, uid_emb)
		out = torch.cat((out, app_top), 1)
		out = self.dec_app(out)
		out = self.dec_app2(out)
		#score = F.selu(out)
		#score = F.tanh(out)
		score = F.sigmoid(out)

		return score

class AppPreUserCGtrHis(nn.Module):
	"""baseline rnn model, only use time, APP usage, with user embedding"""

	def __init__(self, parameters):
		super(AppPreUserCGtrHis, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.his_size = parameters.top_size
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.fc_his = nn.Linear(self.app_emb_size, self.his_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size + self.uid_emb_size + self.his_size, self.app_size)
		
	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim, app_history, tim_history):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
			app_his_emb = app_history.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
			app_his_emb = torch.div(app_history.mm(self.emb_app.weight),torch.sum(app_history,1).view(len(app),1))
			app_his_emb[app_his_emb != app_his_emb] = 0
		app_emb = app_emb.unsqueeze(1)
		
		
		x = torch.cat((tim_emb, app_emb, ptim_emb), 2)
		x = self.dropout(x)
		
		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		app_his = self.fc_his(app_his_emb)
		out = torch.cat((out, uid_emb, app_his), 1)
		out = self.dec_app(out)
		#score = F.selu(out)
		#score = F.tanh(out)
		score = F.sigmoid(out)

		return score

# ############# simple rnn model using location ####################### #
class AppPreLocRUserCGtrHis(nn.Module):
	"""baseline rnn model, considering location with user embedding"""

	def __init__(self, parameters):
		super(AppPreLocRUserCGtrHis, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.his_size = parameters.top_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = 'avg'

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.fc_his = nn.Linear(self.app_emb_size, self.his_size)

		input_size = self.loc_emb_size + self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size + self.uid_emb_size + self.his_size, self.app_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim, app_history, tim_history):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		loc_emb = self.emb_loc(loc)
		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
			app_his_emb = app_history.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
			app_his_emb = torch.div(app_history.mm(self.emb_app.weight),torch.sum(app_history,1).view(len(app),1))
			app_his_emb[app_his_emb != app_his_emb] = 0
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb, loc_emb, ptim_emb), 2)
		x = self.dropout(x)
		
		out, h1 = self.rnn(x, h1)
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(loc.size()[0], 1)
		app_his = self.fc_his(app_his_emb)
		out = torch.cat((out, uid_emb, app_his), 1)

		app_out = self.dec_app(out)
		#app_score = F.tanh(app_out)
		app_score = F.sigmoid(app_out)
		
		return app_score

class AppPreLocRUserCGtr(nn.Module):
	"""baseline rnn model, only use time, APP usage, with user embedding"""

	def __init__(self, parameters):
		super(AppPreLocRUserCGtr, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.loc_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size + self.uid_emb_size, self.app_size)
		
	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		loc_emb = self.emb_loc(loc)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb, ptim_emb, loc_emb), 2)
		x = self.dropout(x)
		
		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		out = torch.cat((out, uid_emb), 1)

		out = self.dec_app(out)
		#score = F.selu(out)
		#score = F.tanh(out)
		score = F.sigmoid(out)

		return score

		
class AppPreLocRUserCGtrTopk(nn.Module):
	"""baseline rnn model, only use time, APP usage, with user embedding"""

	def __init__(self, parameters):
		super(AppPreLocRUserCGtrTopk, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.top_size = parameters.top_size
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.fc_top = nn.Linear(self.app_emb_size, self.top_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.loc_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size + self.uid_emb_size + self.top_size, self.app_size)
		
	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim, app_topk, loc_topk):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		loc_emb = self.emb_loc(loc)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb, ptim_emb, loc_emb), 2)
		x = self.dropout(x)
		
		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		top_emb = app_topk.mm(self.emb_app.weight)
		app_top = self.fc_top(top_emb).repeat(tim.size()[0], 1)
		out = torch.cat((out, uid_emb, app_top), 1)

		out = self.dec_app(out)
		#score = F.selu(out)
		#score = F.tanh(out)
		score = F.sigmoid(out)

		return score

class AppPreLocCUserCGtrHis(nn.Module):
	"""baseline rnn model, considering location with user embedding"""

	def __init__(self, parameters):
		super(AppPreLocCUserCGtrHis, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.his_size = parameters.top_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = 'avg'

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.fc_his = nn.Linear(self.app_emb_size, self.his_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size + self.uid_emb_size + self.his_size + self.loc_emb_size, self.app_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim, app_history, tim_history):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		loc_emb = self.emb_loc(loc)
		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
			app_his_emb = app_history.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
			app_his_emb = torch.div(app_history.mm(self.emb_app.weight),torch.sum(app_history,1).view(len(app),1))
			app_his_emb[app_his_emb != app_his_emb] = 0
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb, ptim_emb), 2)
		x = self.dropout(x)
		
		out, h1 = self.rnn(x, h1)
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(loc.size()[0], 1)
		app_his = self.fc_his(app_his_emb)
		loc_emb = loc_emb.squeeze(1)
		out = torch.cat((out, uid_emb, app_his, loc_emb), 1)

		app_out = self.dec_app(out)
		#app_score = F.tanh(app_out)
		app_score = F.sigmoid(app_out)
		
		return app_score

# ############# rnn model with attention ####################### #
class Attn(nn.Module):
	"""Attention Module. Heavily borrowed from Practical Pytorch
	https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()

		self.method = method
		self.hidden_size = hidden_size

		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, self.hidden_size)
		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
			self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

	def forward(self, out_state, history):
		seq_len = history.size()[0]
		state_len = out_state.size()[0]
		attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
		for i in range(state_len):
			for j in range(seq_len):
				attn_energies[i, j] = self.score(out_state[i], history[j])
		return F.softmax(attn_energies)

	def score(self, hidden, encoder_output):
		if self.method == 'dot':
			energy = hidden.dot(encoder_output)
			return energy
		elif self.method == 'general':
			energy = self.attn(encoder_output)
			energy = hidden.dot(energy)
			return energy
		elif self.method == 'concat':
			energy = self.attn(torch.cat((hidden, encoder_output)))
			energy = self.other.dot(energy)
			return energy

class AppPreUserCGtrHisAttn(nn.Module):
	"""baseline rnn model, only use time, APP usage, with user embedding"""

	def __init__(self, parameters):
		super(AppPreUserCGtrHisAttn, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.his_size = parameters.top_size
		self.attn_type = parameters.attn_type
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size
		self.attn = Attn(self.attn_type, self.hidden_size)
		self.fc_attn = nn.Linear(self.tim_emb_size + self.app_emb_size, self.hidden_size)
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size*2 + self.uid_emb_size, self.app_size)
		
	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim, app_history, tim_history):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		tim_his_emb = self.emb_tim(tim_history)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
			app_his_emb = app_history.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
			app_his_emb = torch.div(app_history.mm(self.emb_app.weight),torch.sum(app_history,1).view(len(app),1))
			app_his_emb[app_his_emb != app_his_emb] = 0
		app_his_emb = app_his_emb.unsqueeze(1)
		history = torch.cat((app_his_emb, tim_his_emb), 2)
		history = F.tanh(self.fc_attn(history))
		history = history.squeeze(1)
		
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb, ptim_emb), 2)
		x = self.dropout(x)
		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		#out = F.selu(out)

		attn_weights = self.attn(out, history).unsqueeze(0)
		context = attn_weights.bmm(history.unsqueeze(0)).squeeze(0)

		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		out = torch.cat((out, context, uid_emb), 1)
		out = self.dec_app(out)
		#score = F.selu(out)
		#score = F.tanh(out)
		score = F.sigmoid(out)

		return score

class AppPreUserPGtrDocAttn(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, parameters):
		super(AppPreUserPGtrDocAttn, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		assert self.app_emb_size==self.uid_emb_size,'app_emb_size!=uid_emb_size,cannot product'
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		self.input_size = self.tim_emb_size + self.app_emb_size
		self.attn_fc = nn.Linear(self.input_size, self.app_emb_size)
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.app_emb_size + self.tim_emb_size, self.app_size)
		
		self.attn_W = nn.Parameter(torch.rand(self.input_size,1))
		self.attn_b = nn.Parameter(torch.rand(4,1))

	def forward(self, tim, app, loc, uid, ptim):
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		out = Variable(torch.rand(ptim.size()[0], self.app_emb_size))
		if self.use_cuda:
			app_emb = app_emb.cuda()
			out = out.cuda()

		tim_emb = self.emb_tim(tim).squeeze(1) #2d-tensor
		ptim_emb = self.emb_tim(ptim).squeeze(1)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb
		
		x = torch.cat((app_emb, tim_emb), 1)
		x = self.dropout(x)

		for i in range (tim.size()[0]-3):
			temp = x[i:i+4,]
			H = F.tanh(torch.add(temp.mm(self.attn_W),self.attn_b))
			X = torch.div(H,torch.sum(torch.abs(H)))
			X = X.view((1,4))
			out[i,:] = self.attn_fc(X.mm(temp))

		uid_emb = self.emb_uid(uid).repeat(ptim.size()[0], 1)
		out = torch.mul(out, uid_emb)
		out = torch.cat((out, ptim_emb), 1)
		out = self.dec_app(out)
		score = F.sigmoid(out)
		return score

# ############# rnn model for location prediction ####################### #
class LocPre(nn.Module):
	"""baseline rnn model, location prediction """

	def __init__(self, parameters):
		super(LocPre, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = 'avg'
		
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
 
		input_size = self.loc_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc = nn.Linear(self.hidden_size, self.loc_size)
		
	def init_weights(self):
		"""
		Here we reproduce Keras default initialization weights for consistency with Keras version
		"""
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()

		loc_emb = self.emb_loc(loc)
		x = self.dropout(loc_emb)

		out, h1 = self.rnn(x, h1)
		out = out.squeeze(1)
		out = F.selu(out)

		y = self.fc(out)
		score = F.log_softmax(y)  # calculate loss by NLLoss
		return score

class LocPreUser(nn.Module):
	"""baseline rnn model, location prediction """

	def __init__(self, parameters):
		super(LocPreUser, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = 'avg'

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

		input_size = self.loc_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc = nn.Linear(self.hidden_size + self.uid_emb_size, self.loc_size)

	def init_weights(self):
		"""
		Here we reproduce Keras default initialization weights for consistency with Keras version
		"""
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()

		loc_emb = self.emb_loc(loc)
		x = self.dropout(loc_emb)

		out, h1 = self.rnn(x, h1)
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(loc.size()[0], 1)
		out = torch.cat((out, uid_emb), 1)

		y = self.fc(out)
		score = F.log_softmax(y)  # calculate loss by NLLoss
		return score

class LocPreGtr(nn.Module):
	"""baseline rnn model, location prediction """

	def __init__(self, parameters):
		super(LocPreGtr, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

		input_size = self.loc_emb_size + self.tim_emb_size*2
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc = nn.Linear(self.hidden_size, self.loc_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()

		loc_emb = self.emb_loc(loc)
		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		x = torch.cat((tim_emb, loc_emb, ptim_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)

		y = self.fc(out)
		score = F.log_softmax(y)  # calculate loss by NLLoss
		return score

class LocPreUserGtr(nn.Module):
	"""baseline rnn model, location prediction """

	def __init__(self, parameters):
		super(LocPreUserGtr, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

		input_size = self.loc_emb_size + self.tim_emb_size*2
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc = nn.Linear(self.hidden_size + self.uid_emb_size, self.loc_size)
	
	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		if self.use_cuda:
			h1 = h1.cuda()

		loc_emb = self.emb_loc(loc)
		ptim_emb = self.emb_tim(ptim)
		tim_emb = self.emb_tim(tim)
		x = torch.cat((tim_emb, loc_emb, ptim_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(loc.size()[0], 1)
		out = torch.cat((out, uid_emb), 1)

		y = self.fc(out)
		score = F.log_softmax(y)  # calculate loss by NLLoss
		return score
	
	def save_emb(self):
		np.save('LocPreUserGtr.npy',self.emb_loc.weight.data.cpu().numpy())
		print('********save************')

class LocPreUserGtrTopk(nn.Module):
	"""baseline rnn model, location prediction """

	def __init__(self, parameters):
		super(LocPreUserGtrTopk, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.top_size = parameters.top_size
		self.rnn_type = parameters.rnn_type

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.fc_top = nn.Linear(self.loc_emb_size, self.top_size)
		
		
		input_size = self.loc_emb_size + self.tim_emb_size*2
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc = nn.Linear(self.hidden_size + self.uid_emb_size + self.top_size*3, self.loc_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim, app_topk, loc_topk):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()

		loc_emb = self.emb_loc(loc)
		ptim_emb = self.emb_tim(ptim)
		tim_emb = self.emb_tim(tim)
		x = torch.cat((tim_emb, loc_emb, ptim_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(loc.size()[0], 1)
		top3_emb = self.emb_loc(loc_topk)
		top3_emb = self.fc_top(top3_emb).view(1, -1).repeat(loc.size()[0], 1)
		out = torch.cat((out, uid_emb, top3_emb), 1)

		y = self.fc(out)
		score = F.log_softmax(y)  # calculate loss by NLLoss
		return score
	
	def save_emb(self):
		np.save('LocPreUserGtrTopk.npy',self.emb_loc.weight.data.cpu().numpy())
		print('********save************')

class LocPreUserGtrRec(nn.Module):
	"""baseline rnn model, location prediction """

	def __init__(self, parameters):
		super(LocPreUserGtrRec, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rec_size = parameters.top_size
		self.rnn_type = parameters.rnn_type

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.rec_loc = nn.Linear(self.loc_emb_size, self.rec_size)
		
		
		input_size = self.loc_emb_size + self.tim_emb_size*2
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc = nn.Linear(self.hidden_size + self.uid_emb_size + self.rec_size, self.loc_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()

		loc_emb = self.emb_loc(loc)
		ptim_emb = self.emb_tim(ptim)
		tim_emb = self.emb_tim(tim)
		x = torch.cat((tim_emb, loc_emb, ptim_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(loc.size()[0], 1)
		rec_emb = self.emb_loc(loc).squeeze(1)
		loc_rec = self.rec_loc(rec_emb)
		out = torch.cat((out, uid_emb, loc_rec), 1)

		y = self.fc(out)
		score = F.log_softmax(y)  # calculate loss by NLLoss
		return score
	
	def save_emb(self):
		np.save('LocPreUserGtrRec.npy',self.emb_loc.weight.data.cpu().numpy())
		print('********save************')

# ############# rnn model for User Identification ####################### #
class UserIden(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(UserIden, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc = nn.Linear(self.hidden_size, self.uid_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb,ptim_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		out = self.fc(out)
		score = F.log_softmax(out)
		
		return score

# #############  Multi-task rnn model ####################### #
class AppPreLocPre(nn.Module):
	"""baseline rnn model, considering location"""

	def __init__(self, parameters):
		super(AppPreLocPre, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = 'avg'

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.loc_emb_size + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		self.fc = nn.Linear(self.hidden_size, self.loc_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		loc_emb = self.emb_loc(loc)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((loc_emb, app_emb), 2)
		x = F.dropout(x)

		out, h1 = self.rnn(x, h1)
		out = out.squeeze(1)
		out = F.selu(out)
		
		app_out = self.dec_app(out)
		#app_score = F.tanh(app_out)
		app_score = F.sigmoid(app_out)
		
		loc_out = self.fc(out)
		loc_score = F.log_softmax(loc_out)  # calculate loss by NLLoss

		return app_score, loc_score

class AppPreLocPreUser(nn.Module):
	"""baseline rnn model, considering location with user embedding"""

	def __init__(self, parameters):
		super(AppPreLocPreUser, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = 'avg'

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.enc_app = nn.Linear(self.app_size, self.app_emb_size)

		input_size = self.loc_emb_size + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		self.fc = nn.Linear(self.hidden_size, self.loc_size)

	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		loc_emb = self.emb_loc(loc)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((loc_emb, app_emb), 2)
		x = self.dropout(x)
		
		out, h1 = self.rnn(x, h1)
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(loc.size()[0], 1)
		out = torch.cat((out, uid_emb), 1)

		app_out = self.dec_app(out)
		#app_score = F.tanh(app_out)
		App_score = F.sigmoid(app_out)
		
		loc_out = self.fc(out)
		loc_score = F.log_softmax(loc_out)  # calculate loss by NLLoss

		return app_score, loc_score

class AppPreLocPreGtr(nn.Module):
	"""baseline rnn model, considering location"""

	def __init__(self, parameters):
		super(AppPreLocPreGtr, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = 'avg'

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.enc_app = nn.Linear(self.app_size, self.app_emb_size)

		input_size = self.loc_emb_size + self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		self.fc = nn.Linear(self.hidden_size, self.loc_size)

	def init_weights(self):
		"""
		Here we reproduce Keras default initialization weights for consistency with Keras version
		"""
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		if self.use_cuda:
			h1 = h1.cuda()

		loc_emb = self.emb_loc(loc)
		ptim_emb = self.emb_tim(ptim)
		tim_emb = self.emb_tim(tim)
		app_enc = self.enc_app(app).unsqueeze(1)
		x = torch.cat((tim_emb, app_enc, loc_emb, ptim_emb), 2)
		x = self.dropout(x)

		out, h1 = self.rnn(x, h1)
		out = out.squeeze(1)
		out = F.selu(out)
		
		app_out = self.dec_app(out)
		#app_score = F.tanh(app_out)
		app_score = F.sigmoid(app_out)
		
		loc_out = self.fc(out)
		loc_score = F.log_softmax(loc_out)  # calculate loss by NLLoss

		return app_score, loc_score

class AppPreLocPreUserGtr(nn.Module):
	"""baseline rnn model, considering location with user embedding"""

	def __init__(self, parameters):
		super(AppPreLocPreUserGtr, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = 'avg'

		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

		input_size = self.loc_emb_size + self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(self.hidden_size + self.uid_emb_size, self.app_size)
		self.fc = nn.Linear(self.hidden_size + self.uid_emb_size, self.loc_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		loc_emb = self.emb_loc(loc)
		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb, loc_emb, ptim_emb), 2)
		x = self.dropout(x)
		
		out, h1 = self.rnn(x, h1)
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(loc.size()[0], 1)
		out = torch.cat((out, uid_emb), 1)

		app_out = self.dec_app(out)
		#app_score = F.tanh(app_out)
		app_score = F.sigmoid(app_out)
		
		loc_out = self.fc(out)
		loc_score = F.log_softmax(loc_out)  # calculate loss by NLLoss

		return app_score, loc_score

class AppPreUserIdenGtr(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(AppPreUserIdenGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc = nn.Linear(self.hidden_size, self.uid_size)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb,ptim_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		app_out = self.dec_app(out)
		app_score = F.sigmoid(app_out)
		
		user_out = self.fc(out)
		user_score = F.log_softmax(user_out)
		
		return app_score,user_score

class AppPreLocPreUserIdenGtr(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(AppPreLocPreUserIdenGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.loc_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc_uid = nn.Linear(self.hidden_size, self.uid_size)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		self.fc_loc = nn.Linear(self.hidden_size, self.loc_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		loc_emb = self.emb_loc(loc)
		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)
		x = torch.cat((tim_emb, app_emb, ptim_emb, loc_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		app_out = self.dec_app(out)
		app_score = F.sigmoid(app_out)
		
		loc_out = self.fc_loc(out)
		loc_score = F.log_softmax(loc_out)  # calculate loss by NLLoss
		
		user_out = self.fc_uid(out)
		user_score = F.log_softmax(user_out)

		return app_score,loc_score,user_score


# ############# Define Loss ####################### #
class AppUserLoss(nn.Module):
	def __init__(self,parameters):
		super(AppUserLoss, self).__init__()
		self.alpha = parameters.loss_alpha
		self.beta = parameters.loss_beta
	def forward(self, app_scores, app_target, uid_scores, uid_target):
		app_loss = nn.BCELoss().cuda()
		uid_loss = nn.NLLLoss().cuda()
		loss_app = app_loss(app_scores, app_target)
		loss_uid = uid_loss(uid_scores, uid_target)
		return loss_app + self.beta*loss_uid, loss_app, loss_uid
	
class AppLocLoss(nn.Module):
	def __init__(self,parameters):
		super(AppLocLoss, self).__init__()
		self.alpha = parameters.loss_alpha
		self.beta = parameters.loss_beta
	def forward(self, app_scores, app_target, loc_scores, loc_target):
		app_loss = nn.BCELoss().cuda()
		loc_loss = nn.NLLLoss().cuda()
		loss_app = app_loss(app_scores, app_target)
		loss_loc = loc_loss(loc_scores, loc_target)
		return loss_app + self.alpha*loss_loc, loss_app, loss_loc

class AppLocUserLoss(nn.Module):
	def __init__(self,parameters):
		super(AppLocUserLoss, self).__init__()
		self.alpha = parameters.loss_alpha
		self.beta = parameters.loss_beta
	def forward(self, app_scores, app_target, loc_scores, loc_target, uid_scores, uid_target):
		app_loss = nn.BCELoss()
		loc_loss = nn.NLLLoss()
		uid_loss = nn.NLLLoss()
		loss_app = app_loss(app_scores, app_target)
		loss_loc = loc_loss(loc_scores, loc_target)
		loss_uid = uid_loss(uid_scores, uid_target)
		return loss_app + self.alpha*loss_loc + self.beta*loss_uid, loss_app, loss_loc, loss_uid

class AppLoss(nn.Module):
	def __init__(self, parameters):
		super(AppLoss, self).__init__()
		self.alpha = parameters.loss_alpha
		self.beta = parameters.loss_beta
	def forward(self, app_scores, app_target, app):
		app_loss1 = nn.BCELoss()
		#app_loss2 = nn.BCELoss()
		#return self.alpha * app_loss2(app_scores, app_target) + (1-self.alpha) * app_loss2(app_scores, app)
		return app_loss1(app_scores, app_target)

class LocLoss(nn.Module):
	def __init__(self, parameters):
		super(LocLoss, self).__init__()
		self.alpha = parameters.loss_alpha
	def forward(self, loc_scores, loc_target, loc):
		loc_loss1 = nn.NLLLoss()
		#loc_loss2 = nn.NLLLoss()
		#return self.alpha * loc_loss2(loc_scores, loc_target) + (1-self.alpha) * loc_loss2(loc_scores, loc)
		return loc_loss1(loc_scores, loc_target)


# ############# Context embedding ####################### #
class Line_1st(nn.Module):
	def __init__(self, num_nodes, emb_size=64):
		super(Line_1st, self).__init__()
		self.order = 1
		self.emb_size = emb_size
		self.num_nodes = num_nodes
		self.emb = nn.Embedding(num_nodes, emb_size)

	def forward(self, x1, x2, w):
		x1 = self.emb(x1)
		x2 = self.emb(x2)
		x = w * torch.sum(x1*x2, dim=1)
		return -F.logsigmoid(x).mean()

	def similarity(self, u, v):
		v1 = self.emb.weight[u]
		v2 = self.emb.weight[v]
		return v1.dot(v2)/(norm(v1)*norm(v2))

class Line_2nd(nn.Module):
	def __init__(self, num_nodes, emb_size=64):
		super(Line_2nd, self).__init__()
		self.order = 2
		self.emb_size = emb_size
		self.num_nodes = num_nodes
		self.emb = nn.Embedding(num_nodes, emb_size)
		self.ctx = nn.Embedding(num_nodes, emb_size) # context vector

	def forward(self, x1, x2, w):
		x1 = self.emb(x1)
		x2 = self.ctx(x2)
		x = w * torch.sum(x1*x2, dim=1)
		return -F.logsigmoid(x).mean()

	def similarity(self, u, v):
		v1 = self.emb.weight[u].data.cpu().numpy()
		v2 = self.emb.weight[v].data.cpu().numpy()
		return v1.dot(v2)/(norm(v1)*norm(v2))

class Line:
	def __init__(self, line_1st, line_2nd, alpha=2, name='0'):
		self.alpha = alpha
		emb1 = line_1st.emb.weight
		emb2 = line_2nd.emb.weight * self.alpha
		self.embedding = torch.cat((emb1, emb2),1)
		self.name = name

	def similarity(self, u, v):
		v1 = self.embedding[u].data.cpu().numpy()
		v2 = self.embedding[v].data.cpu().numpy()
		return v1.dot(v2)/(norm(v1)*norm(v2))

	def save_emb(self):
		np.save('Line_'+self.name+'.npy',self.embedding.data.cpu().numpy())
		print('********save************')

# ############# Joint training ####################### #
class LocPreUserGtrLocEmb(nn.Module):
	"""baseline rnn model, location prediction with LINE as embedding """

	def __init__(self, parameters, line_1st, line_2nd, alpha=2):
		super(LocPreUserGtrLocEmb, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size*2
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda

		self.alpha = alpha
		self.emb_loc1 = line_1st.emb
		self.emb_loc2 = line_2nd.emb
		#self.emb_loc = torch.cat((self.emb_loc1, self.emb_loc2 * self.alpha),1)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

		input_size = self.loc_emb_size + self.tim_emb_size*2
		self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		self.fc = nn.Linear(self.hidden_size + self.uid_emb_size, self.loc_size)

	def forward(self, tim, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		if self.use_cuda:
			h1 = h1.cuda()

		loc_emb1 = self.emb_loc1(loc)
		loc_emb2 = self.emb_loc2(loc)
		loc_emb = torch.cat((loc_emb1, loc_emb2 * self.alpha),2)
		ptim_emb = self.emb_tim(ptim)
		tim_emb = self.emb_tim(tim)
		x = torch.cat((tim_emb, loc_emb), 2)
		x = torch.cat((x, ptim_emb), 2)
		x = F.dropout(x, p=self.dropout_p)

		out, h1 = self.rnn(x, h1)
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(loc.size()[0], 1)
		out = torch.cat((out, uid_emb), 1)

		y = self.fc(out)
		score = F.log_softmax(y)  # calculate loss by NLLoss
		return score
		
	def save_emb(self):
		emb_loc = torch.cat((self.emb_loc1.weight, self.emb_loc2.weight * self.alpha),1)
		np.save('Line_LocPreUserGtrLocEmb.npy',emb_loc.data.cpu().numpy())
		print('********save************')

