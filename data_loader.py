import numpy as np
import math
import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen
import scipy
import scipy.integrate as integrate

import torch
from scipy.stats import norm
from torch.utils.data import DataLoader, Dataset
from scipy.stats import weibull_min


def read_metadata(category):
    prefix = r"C:\Users\march\PycharmProjects\eco-WEU\meta_"
    data = []
    with open(prefix + category + ".json") as f:
        for l in f:
            data.append(json.loads(l.strip()))
    return data



def read_data(category1, category2):
    TrainSamples = convert_Strings_to_Numbers(category1)
    TestSamples = convert_Strings_to_Numbers(category2)
    return TrainSamples, TestSamples

def read_AllSamples(category1, category2):
    meta = read_metadata(category2)
    AllSamples = convert_Strings_to_Numbers(category1)
    items = {}
    with open(category1, 'r+') as y:
        data = y.readlines()
        for line in data:
            currLine = line.split(',')
            items[currLine[0]] = 0
        for i in items:
            for k in meta:
                if 'asin' in k and k['asin'] == i:
                    if 'description' in k and len(k['description']) != 0:
                        if "clean" in k['description'][0]:
                            items[k['asin']] = 1
    return AllSamples, items

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
    distribution = []
    for i in range(len(frequency)):
        if list(frequency[i]).count(0) != 0:
            mu = 0
            for j in range(5):
                mu += (j+1) * frequency[i][j]
            sigma = 0
            for j in range(5):
                sigma += math.pow(j+1-mu,2) * frequency[i][j]
            if sigma == 0:
                sigma = 0.1
            prob_ij = []
            cdf_ij = []
            for r in range(1,5):
                cdf_ij.append(norm.cdf(r+0.5,mu,sigma))
            prob_ij.append(filter(cdf_ij[0]))
            prob_ij.append(filter(cdf_ij[1]-cdf_ij[0]))
            prob_ij.append(filter(cdf_ij[2]-cdf_ij[1]))
            prob_ij.append(filter(cdf_ij[3]-cdf_ij[2]))
            prob_ij.append(filter(1 - cdf_ij[3]))
            distribution.append(prob_ij)
        else:
            distribution.append(list(frequency[i]))
    return np.array(distribution)


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
    itemNum = len(set(Asins))
    userNum = len(set(Reviewers))
    return userNum, itemNum

#def get_price(category):
    #address = "item_price.npy"
    #price = np.load(address)
    #return price

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
    numDict = []
    try:
        with open(category, 'r+') as l:
            data = l.readlines()
            for line in data:
                currLine = line.split(',')
                numDict.append([ReviewersDict[currLine[1]], AsinsDict[currLine[0]], int(float(currLine[2]))])
    except IOError as e:
        print(e)
        print('IOError: Unable to open')
    """print(len(np.unique(Asins)))
    print('--->')
    print(len(np.unique(Reviewers)))
    print('--->')
    print(len(np.unique(Reviews)))
    print('--->')
    print(np.unique(Reviews))"""
    return numDict

class TransactionData(torch.utils.data.Dataset):
    def __init__(self, transactions, userNum, itemNum, trainHist):
        super(TransactionData, self).__init__()
        self.transactions = transactions
        self.L = len(transactions)
        self.user = np.unique(np.array(transactions)[:, 0])
        self.userNum = userNum
        self.itemNum = itemNum
        self.negNum = 2
        self.trainHist = trainHist
        self.userHist = [[] for i in range(self.userNum)]
        for row in transactions:
            self.userHist[row[0]].append(row[1])

    def __len__(self):
        return self.L

    def __getitem__(self, idx):
        row = self.transactions[idx]
        user = row[0]
        item = row[1]
        rating = row[2]
        negItem = self.get_neg(user, item)
        distribution = self.userHist[item]
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


class UserTransactionData(torch.utils.data.Dataset):
    def __init__(self, transactions, userNum, itemNum, trainHist):
        super(UserTransactionData, self).__init__()
        self.transactions = transactions
        self.L = len(transactions)
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
        hist = self.userHist[userId] + self.trainHist[userId]
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

def get_itemDist(AllSamples,ItemNum):
    rating_dist = [[0, 0, 0, 0, 0] for n in range(ItemNum)]
    for element in range(len(AllSamples)):
        item = AllSamples[element][1]
        rating = AllSamples[element][2]
        ratings = []
        for i in range(len(AllSamples)):
            if ((AllSamples[i][1] == item) & (AllSamples[i][2] == rating)):
                rating_dist[item][rating - 1] += 1
        for i in range(len(rating_dist)):
            for k in range(len(rating_dist[i])):
                if (rating_dist[i][k] != 0):
                    ratings.append(k)
    b = 0
    ratings = norm.cdf(ratings)
    for i in range(len(rating_dist)):
        for k in range(len(rating_dist[i])):
            if (rating_dist[i][k] != 0):
                rating_dist[i][k] = ratings[b]
                b+=1
    return rating_dist

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
    train, test = read_data(category1, category2)
    UserNum, ItemNum = get_datasize(catAll)
    AllSamples = read_AllSamples(catAll, 'Appliances')
    frequency = get_itemDist(AllSamples,ItemNum)
    frequency = approx_Gaussian(frequency)

    trainset = TransactionData(train, UserNum, ItemNum, frequency)
    testset = UserTransactionData(test, UserNum, ItemNum, trainset.userHist)
    #comment
    #negItems = trainset.get_neg(train, ItemNum)

    trainset.set_negN(params['negNum_train'])
    trainLoader = DataLoader(trainset, shuffle=True, num_workers=0)

    testset.set_negN(params['negNum_test'])
    testLoader = DataLoader(testset, batch_size=1, num_workers=0)

    for counter, batchData in enumerate(trainLoader):
        if counter == 1:
            break

        users = batchData['user'].numpy().astype(np.int32)

        print('users', type(users))
        print('keys: ', batchData.keys())
        print('r distribution', batchData['r_distribution'])
        print('summation', batchData['r_distribution'].sum(1))
        print('negItem', batchData['negItem'])