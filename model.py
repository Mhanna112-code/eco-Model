import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from heapq import heappush, heappop
from auxiliary import ScaledEmbedding, ZeroEmbedding
import evaluation
import dataloader
from tqdm import tqdm
import numpy as np


class PT(nn.Module):
    def __init__(self, userLen, itemLen, distribution, params, item_price):
        super(PT, self).__init__()
        self.userNum = userLen
        self.itemNum = itemLen
        self.params = params

        if 'gpu' in params and params['gpu'] == True:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        l_size = params['l_size']
        self.distribution = torch.FloatTensor(distribution).to(self.device)
        self.item_price = torch.FloatTensor(item_price).to(self.device)

        self.globalBias_g = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.globalBias_g.weight.data += 0.5
        self.globalBias_g.weight.requires_grad = False
        self.userBias_g = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.itemBias_g = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.userEmbed_g = ScaledEmbedding(userLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_g = ScaledEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.ecoBias_g = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)

        self.globalBias_d = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.globalBias_d.weight.data += 0.5
        self.globalBias_d.weight.requires_grad = False
        self.userBias_d = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.itemBias_d = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.userEmbed_d = ScaledEmbedding(userLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_d = ScaledEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.ecoBias_d = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)

        self.globalBias_a = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.globalBias_a.weight.requires_grad = False
        self.userBias_a = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.userBias_a.weight.data.uniform_(0.0, 0.05)
        self.itemBias_a = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.itemBias_a.weight.data.uniform_(0.0, 0.05)
        self.ecoBias_a = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.ecoBias_a.weight.data.uniform_(0.0, 0.05)
        self.userEmbed_a = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
        self.userEmbed_a.weight.data.uniform_(-0.01, 0.01)
        self.itemEmbed_a = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_a.weight.data.uniform_(-0.01, 0.01)

        self.globalBias_b = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.globalBias_b.weight.requires_grad = False
        self.ecoBias_b = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.ecoBias_b.weight.data.uniform_(0.0, 0.05)
        self.userBias_b = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.userBias_b.weight.data.uniform_(0.0, 0.05)
        self.itemBias_b = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.itemBias_b.weight.data.uniform_(0.0, 0.05)
        self.userEmbed_b = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
        self.userEmbed_b.weight.data.uniform_(-0.01, 0.01)
        self.itemEmbed_b = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_b.weight.data.uniform_(-0.01, 0.01)

        self.globalBias_l = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.globalBias_l.weight.data += 1
        self.globalBias_l.weight.requires_grad = False
        self.userBias_l = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.userBias_l.weight.data.uniform_(0.0, 0.05)
        self.itemBias_l = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.ecoBias_l = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.ecoBias_l.weight.data.uniform_(0.0, 0.05)
        self.itemBias_l.weight.data.uniform_(0.0, 0.05)
        self.userEmbed_l = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
        self.userEmbed_l.weight.data.uniform_(-0.01, 0.01)
        self.itemEmbed_l = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_l.weight.data.uniform_(-0.01, 0.01)

        self.reference_point = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        print("r = ")
        print(self.reference_point)
        self.reference_point.weight.data = torch.ones_like(self.reference_point.weight.data) * 1.5
        #         self.reference_point.weight.requires_grad=False
        self.to(self.device)
        self.grads = {}

    def forward(self, users, items):
        distribution = self.distribution[items].to(self.device)
        reference_point = self.reference_point(torch.tensor(0).to(self.device))
        #         print(users.shape[0],items.shape[0])
        price = self.item_price[items].view(-1,1).expand(users.shape[1],397).to(self.device)

        # calculate value
        globalBias_a = self.globalBias_a(torch.tensor(0).to(self.device))
        print("gB = ")
        print(globalBias_a)
        userBias_a = self.userBias_a(torch.tensor(0).to(self.device))
        itemBias_a = self.itemBias_a(torch.tensor(0).to(self.device))
        ecoBias_a = self.ecoBias_a(torch.tensor(0).to(self.device))
        userEmbed_a = [[0 for n in range(userNum)] for n in range(params['l_size'])]
        userEmbed_a = np.transpose(userEmbed_a)
        userEmbed_a = torch.FloatTensor(userEmbed_a)
        itemEmbed_a = [[0 for n in range(itemNum)] for n in range(params['l_size'])]
        itemEmbed_a = torch.FloatTensor(itemEmbed_a)

        globalBias_b = self.globalBias_b(torch.tensor(0).to(self.device))
        userBias_b = self.userBias_b(torch.tensor(0).to(self.device))
        itemBias_b = self.itemBias_b(torch.tensor(0).to(self.device))
        userEmbed_b = [[0 for n in range(userNum)] for n in range(params['l_size'])]
        userEmbed_b = np.transpose(userEmbed_b)
        userEmbed_b = torch.FloatTensor(userEmbed_b)
        itemEmbed_b = [[0 for n in range(itemNum)] for n in range(params['l_size'])]
        itemEmbed_b = torch.FloatTensor(itemEmbed_b)
        ecoBias_b= self.ecoBias_a(torch.tensor(0).to(self.device))

        globalBias_l = self.globalBias_l(torch.tensor(0).to(self.device))
        userBias_l = self.userBias_l(torch.tensor(0).to(self.device))
        itemBias_l = self.itemBias_l(torch.tensor(0).to(self.device))
        ecoBias_l = self.ecoBias_l(torch.tensor(0).to(self.device))
        userEmbed_l = [[0 for n in range(userNum)] for n in range(params['l_size'])]
        userEmbed_l = np.transpose(userEmbed_l)
        userEmbed_l = torch.FloatTensor(userEmbed_l)
        itemEmbed_l = [[0 for n in range(itemNum)] for n in range(params['l_size'])]
        itemEmbed_l = torch.FloatTensor(itemEmbed_l)

        alpha = globalBias_a + userBias_a + ecoBias_a + itemBias_a + torch.matmul(userEmbed_a, itemEmbed_a).sum(1).view(-1, 1)
        beta = globalBias_b + userBias_b + ecoBias_b + itemBias_b + torch.matmul(userEmbed_b, itemEmbed_b).sum(1).view(-1, 1)
        #need delta, gamma + theta
        lamda = globalBias_l + userBias_l + ecoBias_l + itemBias_l + torch.matmul(userEmbed_l, itemEmbed_l).sum(1).view(-1, 1)

        rating = torch.tensor([1., 2., 3., 4., 5.]).expand(users.shape[1], 397).to(self.device)
        x = torch.tanh(rating - reference_point)
        x_binary_pos = torch.gt(x, torch.FloatTensor([0]).to(self.device)).to(torch.float)
        x_binary_neg = torch.ones_like(x).to(self.device) - x_binary_pos

        x_ = torch.mul(price, torch.abs(x))
        v_exp = torch.mul(alpha, x_binary_pos) + torch.mul(beta, x_binary_neg)
        v = x_.pow(v_exp)
        v_coef = x_binary_pos - torch.mul(lamda, x_binary_neg)
        value = torch.mul(v, v_coef).to(self.device)

        # calculate weight
        globalBias_g = self.globalBias_g(torch.tensor(0).to(self.device))
        userBias_g = self.userBias_g(torch.tensor(0).to(self.device))
        itemBias_g = self.itemBias_g(torch.tensor(0).to(self.device))
        ecoBias_g = self.ecoBias_g(torch.tensor(0).to(self.device))
        itemEmbed_g = [[0 for n in range(itemNum)] for n in range(params['l_size'])]
        itemEmbed_g = torch.FloatTensor(itemEmbed_g)
        userEmbed_g = [[0 for n in range(userNum)] for n in range(params['l_size'])]
        userEmbed_g = np.transpose(userEmbed_g)
        userEmbed_g = torch.FloatTensor(userEmbed_g)

        globalBias_d = self.globalBias_d(torch.tensor(0).to(self.device))
        ecoBias_d = self.ecoBias_d(torch.tensor(0).to(self.device))
        userBias_d = self.userBias_d(torch.tensor(0).to(self.device))
        itemBias_d = self.itemBias_d(torch.tensor(0).to(self.device))
        userEmbed_d = [[0 for n in range(userNum)] for n in range(params['l_size'])]
        userEmbed_d = np.transpose(userEmbed_d)
        userEmbed_d = torch.FloatTensor(userEmbed_d)
        itemEmbed_d = [[0 for n in range(itemNum)] for n in range(params['l_size'])]
        itemEmbed_d = torch.FloatTensor(itemEmbed_d)

        numitemEm = np.array(itemEmbed_g)
        numuserEm = np.array(userEmbed_g)
        print("======")
        print(numitemEm.shape)
        print(numuserEm.shape)

        gamma = ecoBias_g + globalBias_g + userBias_g + itemBias_g + torch.matmul(userEmbed_g, itemEmbed_g).sum(1).view(-1, 1)
        delta = ecoBias_d + globalBias_d + userBias_d + itemBias_d + torch.matmul(userEmbed_d, itemEmbed_d).sum(1).view(-1, 1)

        gamma_ = gamma.expand(users.shape[1], 397)
        delta_ = delta.expand(users.shape[1], 397)
        w_exp = torch.mul(x_binary_pos, gamma_) + torch.mul(x_binary_neg, delta_)

        w_nominator = distribution.pow(w_exp)
        w_denominator = (distribution.pow(w_exp) + (torch.ones_like(distribution).to(self.device) - distribution).pow(
            w_exp)).pow(1 / w_exp)
        weight = torch.div(w_nominator, w_denominator)

        #         self.userBias_g.weight.register_hook(self.save_grad('userBias_g'))
        #         self.itemBias_g.weight.register_hook(self.save_grad('itemBias_g'))
        #         self.userEmbed_g.weight.register_hook(self.save_grad('userEmbed_g'))
        #         self.itemEmbed_g.weight.register_hook(self.save_grad('itemEmbed_g'))
        return torch.mul(weight, value).sum(1)

    def loss(self, users, items, negItems):
        users = users.expand(users.shape[0], self.params['negNum_train']).to(self.device)

        pOut = self.forward(users, items).view(-1, 1).expand(users.shape[0], self.params['negNum_train']).reshape(-1, 1)
        nOut = self.forward(users, negItems).reshape(-1, 1)

        criterion = nn.Sigmoid()
        loss = torch.mean(criterion(pOut - nOut))
        return -loss

    def get_grads(self):
        return self.grads

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad

        return hook


if __name__ == '__main__':
    keys_file = open("S&O.json", 'r', errors='ignore', encoding='UTF-8')
    params = dict()
    params['lr'] = 1e-3
    params['batch_size'] = 2
    params['epoch_limit'] = 10
    params['w_decay'] = 1.
    params['negNum_test'] = 1000
    params['epsilon'] = 1e-4
    params['negNum_train'] = 2
    params['l_size'] = 16
    params['train_device'] = 'cpu'
    params['test_device'] = 'cpu'
    params['lambda'] = 1.
    params['test_per_train'] = 2

    category1 = 'newTrainSamples'
    category2 = 'newTestSamples'
    catAll = 'AllSamples'
    train, test = dataloader.read_data(category1,category2)
    ratings = []
    for rating in train:
        ratings.append(train[rating][1])
    data = np.array(ratings)
    trainUserNum, trainItemNum = dataloader.get_datasize(category1)
    testUserNum, testItemNum = dataloader.get_datasize(category2)
    frequency = dataloader.get_distribution(data)
    distribution = dataloader.approx_Gaussian(frequency)

    trainset = dataloader.TransactionData(train, trainUserNum, trainItemNum, distribution)
    trainLoader = DataLoader(trainset, shuffle=False, num_workers=0)

    testset = dataloader.UserTransactionData(test, testUserNum, testItemNum, trainset.userHist)
    testset.set_negN(params['negNum_test'])
    testLoader = DataLoader(testset, shuffle=False, num_workers=0)

    prices = np.full(11,100)
    model = PT(userLen=trainUserNum, itemLen=testItemNum, distribution=distribution, params=params, item_price=prices)
    print('initialization', model.state_dict())
    optimizer = optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['w_decay'])

    epoch = 0
    print('start training...')
    while epoch < params['epoch_limit']:
        model.device = params['train_device']
        model.to(model.device)

        epoch += 1
        print('Epoch ', str(epoch), ' training...')
        L = len(trainLoader.dataset)
        pbar = tqdm(total=L)
        for i, batchData in enumerate(trainLoader):
            optimizer.zero_grad()
            users = torch.LongTensor(batchData['user']).to(model.device)
            items = torch.LongTensor(batchData['item']).to(model.device)
            negItems = torch.LongTensor(batchData['negItem']).reshape(-1).to(model.device)

            batch_loss = model.loss(users, items, negItems)
            batch_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if i == 0:
                total_loss = batch_loss.clone()
            else:
                total_loss += batch_loss.clone()
            pbar.update(users.shape[0])
        pbar.close()
        # torch.save(model, 'pt.pt')
        print('epoch loss', total_loss)
        print(model.state_dict())

        if epoch % params['test_per_train'] == 0:
            print('starting test...')
            model.device = params['test_device']
            model.to(model.device)
            L = len(testLoader.dataset)
            pbar = tqdm(total=L)
            with torch.no_grad():
                scoreDict = dict()
                for i, batchData in enumerate(testLoader):
                    user = torch.LongTensor(batchData['user']).to(model.device)
                    posItems = torch.LongTensor(batchData['posItem']).to(model.device)
                    negItems = torch.LongTensor(batchData['negItem']).to(model.device)

                    items = torch.cat((posItems, negItems), 1).view(-1)
                    users = user.expand(items.shape[0])

                    score = model.forward(users, items)
                    scoreHeap = list()
                    for j in range(score.shape[0]):
                        gt = False
                        if j < posItems.shape[1]:
                            gt = True

                        heappush(scoreHeap, (1 - score[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
                    scores = list()
                    candidate = len(scoreHeap)
                    for k in range(candidate):
                        scores.append(heappop(scoreHeap))
                    pbar.update(1)
                    scoreDict[user[0]] = (scores, posItems.shape[1])
            pbar.close()
            testResult = evaluation.ranking_performance(scoreDict, 100)