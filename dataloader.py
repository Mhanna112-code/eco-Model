import numpy as np
import math
from scipy.stats import norm
import torch
from torch.utils.data import DataLoader, Dataset
from collections import namedtuple

def read_data(category1, category2):
	TrainSamples = {}
	TestSamples = {}
	TrainSamples = convert_Strings_to_Numbers(category1)
	TestSamples = convert_Strings_to_Numbers(category2)
	return TrainSamples, TestSamples


# def convert(samples):
# 	# input one of [TrainSamples, ValSamples, newTestSamples]
# 	length = len(samples)
# 	user_ids = []
# 	item_ids = []
# 	ratings = []
# 	for transaction in samples:
# 		user_ids.append(transaction[0])
# 		item_ids.append(transaction[1])
# 		ratings.append(transaction[2])
# 	user_ids = np.array(user_ids)
# 	user_ids.astype(np.int32)
# 	item_ids = np.array(item_ids)
# 	item_ids.astype(np.int32)
# 	ratings = np.array(ratings)
# 	ratings.astype(np.int32)
# 	# return type doesn't support index
# 	return interactions.Interactions(user_ids, item_ids, ratings)


def approx_Gaussian(frequency):
#	distribution = []
#	for i in range(len(frequency)):
#		mu = 0
#		for j in range(5):
#			mu += (j+1) * frequency[i][j]
#		sigma = 0
#		for j in range(5):
#			sigma += math.pow(j+1-mu,2) * frequency[i][j]
#		if sigma == 0:			sigma = 0.1
#		prob_ij = []
#		cdf_ij = []
#		for r in range(1,5):
#			cdf_ij.append(norm.cdf(r+0.5,mu,sigma))
#		prob_ij.append(filter(cdf_ij[0]))
#		prob_ij.append(filter(cdf_ij[1]-cdf_ij[0]))
#		prob_ij.append(filter(cdf_ij[2]-cdf_ij[1]))
#		prob_ij.append(filter(cdf_ij[3]-cdf_ij[2]))
#		prob_ij.append(filter(1 - cdf_ij[3]))
#		distribution.append(prob_ij)
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
	AllSamples = {}
	try:
		f = open(category, 'r')
		for line in f:
			CustomerData = namedtuple("Reviewer", ["item", "Rating"])
			item, reviewerID, rating = line.split(',')
			AllSamples[reviewerID] = CustomerData(item, rating)
	except IOError as e:
		print(e)
		print('IOError: Unable to open')
	reviewers = AllSamples.keys()
	intRs = []
	for reviewer in reviewers:
		intRs.append(int.from_bytes(reviewer.encode(),'little'))
	items = []
	for reviewer in AllSamples.values():
		items.append(reviewer[0])
	npReviewers = np.array(intRs)
	npItems = np.array(items)
	userNum = len(np.unique(npReviewers))
	itemNum = len(np.unique(npItems))
	return userNum, itemNum

#def get_price(category):
	#address = "item_price.npy"
	#price = np.load(address)
	#return price


def get_distribution(category):
	address = "ItemResult.npy"
	distribution = np.load(address)
	return distribution

def convert_Strings_to_Numbers(category):
	Asins = []
	Reviewers = []
	Reviews = []
	try:
		with open(category, 'r+') as y:
			data = y.readlines()
			for line in data:
				currLine = line.split(',')
				Asins.append(currLine[0])
				Reviewers.append(currLine[1])
				Reviews.append(currLine[2])
	except IOError as e:
		print(e)
		print('IOError: Unable to open')
	UnqAsins = set(Asins)
	UnqReviewers = set(Reviewers)
	idx = 0
	AsinsDict = {}
	for i in UnqAsins:
		AsinsDict[i] = idx
		idx += 1
	idx = 0
	ReviewersDict = {}
	for i in UnqReviewers:
		ReviewersDict[i] = idx
		idx += 1
	numDict = {}
	try:
		with open(category, 'r+') as l:
			data = l.readlines()
			for line in data:
				currLine = line.split(',')
				numDict[ReviewersDict[currLine[1]]] = [AsinsDict[currLine[0]], int(float(currLine[2]))]
	except IOError as e:
		print(e)
		print('IOError: Unable to open')
	print(len(np.unique(Asins)))
	print('--->')
	print(len(np.unique(Reviewers)))
	print('--->')
	print(len(np.unique(Reviews)))
	print('--->')
	print(np.unique(Reviews))
	return numDict

class TransactionData(torch.utils.data.Dataset):
	def __init__(self, transactions, userNum, itemNum, rating_distribution):
		super(TransactionData, self).__init__()
		self.transactions = transactions
		self.L = len(transactions)
		self.users = np.unique(np.array(transactions.keys()))
		self.userNum = userNum
		self.itemNum = itemNum
		self.negNum = 2
		self.rating_distribution = rating_distribution
		self.userHist = [[] for i in range(self.userNum)]
		userList = []
		itemList = []
		for row in transactions:
			itemList.append(transactions[row][0])
			userList.append(row)
		self.userHist = [[] for i in range(self.itemNum)]
		for i in range(len(itemList)):
			self.userHist[itemList[i]].append(userList[i])
		#self.userHist = np.array(self.userHist)
		#self.userHist = torch.from_numpy(self.userHist[:])
	#comment
	def __len__(self):
		return self.L

	def __getitem__(self, idx):
		if idx > 10:
			idx = 0
		user = self.userHist[idx]
		item = idx
		rating = []
		for reviewer in self.transactions:
			if int(self.transactions[reviewer][0]) == idx:
				rating.append(int(float(self.transactions[reviewer][1])))
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
		hist = []
		for user in self.userHist:
			if user == userid:
				hist.append(userid)
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


class UserTransactionData(torch.utils.data.Dataset):
	def __init__(self, transactions, userNum, itemNum, trainHist):
		super(UserTransactionData, self).__init__()
		self.transactions = transactions
		self.L = userNum
		self.users = np.unique(np.array(transactions.keys()))
		self.userNum = userNum
		self.itemNum = itemNum
		self.negNum = 10
		self.userHist = [[] for i in range(self.itemNum)]
		self.trainHist = trainHist
		userList = []
		itemList = []
		for row in transactions:
			itemList.append(transactions[row][0])
			userList.append(row)
		self.userHist = [[] for i in range(self.itemNum)]
		for i in range(len(itemList)):
			self.userHist[itemList[i]].append(userList[i])
		#self.userHist = np.array(self.userHist)
		#self.userHist = torch.from_numpy(self.userHist[:])

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
	category1 = 'newTrainSamples'
	category2 = 'newTestSamples'
	train = read_data(category1, category2)[0]
	catAll = 'AllSamples'
	params = dict()
	params['batch_size'] = 3
	params['epoch_limit'] = 1
	params['w_decay'] = 1
	params['negNum_test'] = 1000
	params['negNum_train'] = 2
	params['l_size'] = 11
	reviews = []
	for reviewer in train.values():
		reviews.append(reviewer[1])
	data = np.array(reviews)
	np.save('ItemResult', data)
	train, test = read_data(category1, category2)
	userNum, itemNum = get_datasize(catAll)
	frequency = get_distribution(catAll)
	distribution = norm.cdf(frequency)
	trainset = TransactionData(train, userNum, itemNum, distribution)
	testset = UserTransactionData(test, userNum, itemNum, trainset.userHist)
	#comment
	trainset.set_negN(params['negNum_train'])
	trainLoader = DataLoader(trainset)

	testset.set_negN(params['negNum_test'])
	#testLoader = DataLoader(testset, batch_size=1, num_workers=0)

	for counter, batchData in enumerate(trainLoader):
		print(batchData)
		if counter == 1:
			break

		users = batchData['user'].numpy().astype(np.int32)
		#print('users', type(users))
		#print('keys: ', batchData.keys())
		#print('r distribution', batchData['r_distribution'])
		#print('summation', batchData['r_distribution'].sum(1))
		#print('negItem', batchData['negItem'])

