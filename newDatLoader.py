from importlib._common import _

import numpy as np
import math

from scipy.stats import norm
import copy
from torch.utils.data import Dataset, DataLoader
import torch

import os

prefix = '../data/'

def read_data(category):
	#with open('TrainSamples', 'r') as f:
	#	data = f.readlines()
	#TrainSamples = []
	#for line in data:
	#	row = line[:-1].split(',')
	#	sample=[int(float(i)) for i in row]
	#	TrainSamples.append(sample)
	i = 0
	# with open(address+'ValidationSamples.txt', 'r') as f:
	# 	data = f.readlines()
	# ValSamples = []
	# for line in data:
	# 	row = line[:-1].split(',')
	# 	sample = [int(float(i)) for i in row]
	# 	ValSamples.append(sample)
	with open(category, 'r') as f:
		data = f.readlines()
	TestSamples = []
	for line in data:
		row = line.split(',')
		TestSamples.append(row)
	Asins,Reviewers,Reviews = convert_Strings_to_Numbers(category, TestSamples)
	for n in TestSamples:
		n[0] = Asins[i]
		n[1] = Reviewers[i]
		n[2] = Reviews[i]
		i += 1
	return TestSamples

def convert_Strings_to_Numbers(category, *TestSamples):
	Asins = []
	Reviewers = []
	Reviews = []
	with open(category, 'r+') as y:
		data = y.readlines()
	for line in data:
		currLine = line.split(',')
		Asins.append(currLine[0])
		Reviewers.append(currLine[1])
		Reviews.append(currLine[2])
	print(len(np.unique(Asins)))
	print('--->')
	print(len(np.unique(Reviewers)))
	print('--->')
	print(len(np.unique(Reviews)))
	print('--->')
	print(np.unique(Reviews))
	print('============================')
	print(category)
	#Asins = [math.floor(int.from_bytes(n.encode(), 'little')/10000000000000000) for n in Asins]
	#Reviewers = [math.floor(int.from_bytes(n.encode(), 'little')/100000000000000000000000000) for n in Reviewers]
	#Reviews = [int(float(n)) for n in Reviews]
	UnqAsins = set(Asins)
	UnqReviewers = set(Reviewers)
	idx = 0
	for i in Asins:
		for j in UnqAsins:
			if i == j:
				continue
			else:
				i = idx
				idx += 1
	print(len(np.unique(Asins)))
	print('--->')
	print(len(np.unique(Reviewers)))
	print('--->')
	print(len(np.unique(Reviews)))
	print('--->')
	print(np.unique(Reviews))
	return Asins,Reviewers,Reviews

def approx_Gaussian(frequency):

	distribution = norm.cdf(frequency)
	return np.array(distribution)

# def approx_Gaussian(frequency):
# 	distribution = []
# 	for i in range(len(frequency)):
# 		if list(frequency[i]).count(0) != 0:
# 			mu = 0
# 			for j in range(5):
# 				mu += (j+1) * frequency[i][j]
# 			sigma = 0
# 			for j in range(5):
# 				sigma += math.pow(j+1-mu,2) * frequency[i][j]
# 			if sigma == 0:
# 				sigma = 0.1
# 			prob_ij = []
# 			cdf_ij = []
# 			for r in range(1,5):
# 				cdf_ij.append(norm.cdf(r+0.5,mu,sigma))
# 			prob_ij.append(filter(cdf_ij[0]))
# 			prob_ij.append(filter(cdf_ij[1]-cdf_ij[0]))
# 			prob_ij.append(filter(cdf_ij[2]-cdf_ij[1]))
# 			prob_ij.append(filter(cdf_ij[3]-cdf_ij[2]))
# 			prob_ij.append(filter(1 - cdf_ij[3]))
# 			distribution.append(prob_ij)
# 		else:
# 			distribution.append(list(frequency[i]))
# 	return np.array(distribution)


def filter(prob):
	if prob <= 1e-4:
		return 1e-4
	elif prob >= 1-1e-4:
		return 1-1e-4
	else:
		return prob


def get_decumulative(distribution):
	decumulative = [[1.0] for i in range(distribution.shape[0])]
	# decumulative = copy.deepcopy(cumulative)
	for i in range(distribution.shape[0]):
		distribution_i = distribution[i]
		# print('distribution', distribution_i)
		# decumulative[i].append(1.0)
		for j in range(1, 6):
			summation = sum(distribution_i[:j])
			if summation >= 1.:
				decumulative[i].append(1e-10)
			elif summation <= 1e-10:
				decumulative[i].append(1.0)
			else:
				decumulative[i].append(1.-summation)
	return np.array(decumulative)


def get_datasize(category):
	AllSamples = list()
	with open(category,'r') as f:
		data = f.readlines()
	for line in data:
		row = line.rstrip().split(',')
		AllSamples.append(row)
	all_data = np.array(AllSamples)
	userNum = len(np.unique(all_data[:,1]))
	itemNum = len(np.unique(all_data[:,0]))
	return userNum, itemNum

def get_price(category):
	address = prefix + category + "/" + category + "_" + "item_price.npy"
	price = np.load(address)
	return price

def get_distribution(category):
	distribution = np.loadtxt(category,encoding='utf-8')
	return distribution


class TransactionData(Dataset):
	def __init__(self, transactions, userNum, itemNum, rating_distribution):
		super(TransactionData, self).__init__()
		self.transactions = transactions
		self.L = len(transactions)
		self.users = np.unique(np.array(transactions)[:, 1])
		self.userNum = userNum
		self.itemNum = itemNum
		self.negNum = 2
		self.rating_distribution = rating_distribution
		self.userHist = [[] for i in range(self.userNum)]
		for row in transactions:
			self.userHist[row[1]%userNum].append(row[0])

	def __len__(self):
		return self.L

	def __getitem__(self, idx):
		row = self.transactions[idx]
		user = row[0]
		item = row[1]
		rating = row[2]
		negItem = self.get_neg(user, item)
		distribution = self.rating_distribution[item]
		return {'user': np.array(user).astype(np.int64),
				'item': np.array(item).astype(np.int64),
				'r_distribution': np.array(distribution).astype(float),
				'rating': np.array(rating).astype(float),
				'negItem': np.array(negItem).astype(np.int64)
				}

	def get_neg(self, userid, itemid):
		neg = list()
		hist = self.userHist[userid]
		for i in range(self.negNum):
			while True:
				negId = np.random.randint(self.itemNum)
				if negId not in hist and negId not in neg:
					neg.append(negId)
					break
		return neg

	def set_negN(self, n):
		if n < 1:
			return
		self.negNum = n


class UserTransactionData(Dataset):
	def __init__(self, transactions, userNum, itemNum, trainHist):
		super(UserTransactionData, self).__init__()
		self.transactions = transactions
		self.L = userNum
		self.user = np.unique(np.array(transactions)[:, 0])
		self.userNum = userNum
		self.itemNum = itemNum
		self.negNum = 10
		self.userHist = [[] for i in range(self.userNum)]
		self.trainHist = trainHist
		for row in transactions:
			self.userHist[row[0]].append(row[1])

	def __len__(self):
		return self.L

	def __getitem__(self, idx):
		user = self.user[idx]
		posItem = self.userHist[idx]

		negPrice = []
		negItem = self.get_neg(idx)

		return {'user': np.array(user).astype(np.int64),
				'posItem': np.array(posItem).astype(np.int64),
				'negItem': np.array(negItem).astype(np.int64)
				}

	def get_neg(self, userId):
		hist = self.userHist[userId]+self.trainHist[userId]
		neg = []
		for i in range(self.negNum):
			while True:
				negId = np.random.randint(self.itemNum)
				if negId not in hist and negId not in neg:
					neg.append(negId)
					break
		return neg

	def set_negN(self, n):
		if n < 1:
			return
		self.negNum = n


if __name__ == '__main__':
	# train = read_data('Movielens')[0]
	# print(train[1])
	# inter = convert(train)
	# print(type(inter))


	category1 = 'newTrainSamples'
	category2 = 'newTestSamples'
	catAll = 'AllSamples'
	params = dict()
	params['batch_size'] = 32
	params['epoch_limit'] = 1
	params['w_decay'] = 1
	params['negNum_test'] = 1000
	params['negNum_train'] = 2
	params['l_size'] = 16

	train = read_data(category1)
	test = read_data(category2)
	userNum, itemNum = get_datasize(catAll)
	#np.savetxt('AllSamples.npy', read_data(catAll))
	frequency = np.array(read_data(catAll))
	distribution = approx_Gaussian(frequency)

	trainset = TransactionData(train, userNum, itemNum, distribution)
	testset = UserTransactionData(test, userNum, itemNum, distribution)

	trainset.set_negN(params['negNum_train'])
	trainLoader = DataLoader(trainset, batch_size = params['batch_size'], shuffle=True, num_workers=0)

	testset.set_negN(params['negNum_test'])
	testLoader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)

	for counter, batchData in enumerate(trainLoader):
		if counter == 1:
			break

		users = batchData['user'].numpy().astype(np.int32)
		print('users', type(users))
		print('keys: ', batchData.keys())
		print('r distribution', batchData['r_distribution'])
		print('summation', batchData['r_distribution'].sum(1))
		print('negItem', batchData['negItem'])