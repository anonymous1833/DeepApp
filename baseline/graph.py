'''
	Created on Jan 23, 2016
	Handle all the data preprocessing, turning files to numpy files
	Sample graph to generate labels

	@author: Lanxiao Bai, Carl Yang
'''
import numpy as np
import ast
import random
import pickle
from scipy.sparse import csr_matrix
from math import sqrt
import bisect
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import torch
random.seed(2)

def weight_choice(weight):
	"""
	:param weight: 
	:return: index sort by weight list
	"""
	weight_sum = []
	sum = 0
	for a in weight:
		sum += a
		weight_sum.append(sum)
	t = random.randint(0, sum - 1)
	return bisect.bisect_right(weight_sum, t)
	
def distance(a, b):
	return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)  
	
class Graph(object):
	def __init__(self, num_walks=10, length_walk=30, window_size=5):
		self.length_walk = length_walk
		self.num_walks = num_walks
		self.window_size = window_size

	def getSpotGraph(self,coordinates,sample_portion = 0.1,sample_radius = 0.05):
		#coordinates[spot_idx] = (float(x), float(y))
		relation_dict = {}
		spot_enum = len(coordinates)
		density = 0
		sample_size = int(spot_enum*sample_portion)
		#print('Orignal '+str(spot_enum)+' base spots to build spot graph')
		#print('Sampling '+str(sample_size)+' base spots to build spot graph')
		base_points = random.sample(coordinates.keys(), k=sample_size)
		for base in base_points:
			cell = []
			for i in coordinates.keys():
				if distance(coordinates[i], coordinates[base]) < sample_radius:
					cell.append(i)
			#print('Cell '+str(base)+' has '+str(len(cell))+' spots')
			for i in cell:
				for j in cell:
					if i != j:
						if i not in relation_dict:
							relation_dict[i] = set()
						if j not in relation_dict:
							relation_dict[j] = set()
						relation_dict[i].add(j)
						relation_dict[j].add(i)

		for i in relation_dict.keys():
			relation_dict[i] = list(relation_dict[i])
			density += len(relation_dict[i])
		density = density * 1.0 / (spot_enum*spot_enum)
		#print('Density of spot graph: '+str(density))
		#print('relation_dict: ', len(relation_dict.keys()))
		
		edge = []
		for i in range (len(coordinates)):
			if i in relation_dict:
				for j in relation_dict[i]:
					edge.append([i,j])
		loc_edge = np.array(edge)
		#print(loc_edge[0,:])
		#print(loc_edge.shape)
		
		G = nx.Graph()
		for i in range(len(coordinates)):
			G.add_node(i)
		G.add_edges_from(loc_edge)
		
		return G

	

class NegativeSampler:
	def __init__(self, lengths, factor_M=10):
		self.lengths = np.cumsum(lengths) #CDF
		self.M = len(lengths) * factor_M
		self.table = self.creat_table()

	def fetch(self, u, v):
		while True:
			m = random.randint(0, self.M-1)
			node = self.table[m]
			if node != u and node != v:
				break

		return node

	def creat_table(self):
		table = []
		idx = 0
		ms = np.linspace(0, 1, self.M)
		for m in ms:
			if m >= self.lengths[idx]:
				if (idx<len(self.lengths)-1): #limited the index
					idx += 1
			table.append(idx)
		print('**********Alian Table***********')
		print(len(table))
		return table

class SmapleDataset(Dataset):
	def __init__(self, G, neg_sample_size=5):
		self.G = G
		self.edges = list(self.G.edges())
		self.neg_size = neg_sample_size
		if self.neg_size > 0:
			self.order = 2
			lengths = np.asarray([G.degree(node)**0.75 for node in G.nodes()])
			lengths = lengths / np.sum(lengths) #the probability,pdf
			self.neg_smapler = NegativeSampler(lengths)
		else:
			self.order = 1

	def __len__(self):
		return self.G.number_of_edges()

	def __getitem__(self, idx):
		u, v = self.edges[idx]
		if self.order == 1:
			return torch.LongTensor([np.int64(u)]), torch.LongTensor([np.int64(v)]), torch.FloatTensor([1.0])

		us, vs, ws = [np.int64(u)], [np.int64(v)], [1.0]
		for _ in range(self.neg_size):
			v = self.neg_smapler.fetch(u, v)
			us.append(np.int64(u))
			vs.append(np.int64(v))
			ws.append(-1.0)
		return torch.LongTensor(us), torch.LongTensor(vs), torch.FloatTensor(ws)
	