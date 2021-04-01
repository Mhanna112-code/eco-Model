import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pdb
import matplotlib as plt
from heapq import heappush, heappop
from auxiliary import ScaledEmbedding, ZeroEmbedding
import evaluation
import data_loader
from tqdm import tqdm
from scipy.special import expit
import sklearn
import math
import time

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
        self.ecoBias_g = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.userBias_g = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.itemBias_g = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.userEmbed_g = ScaledEmbedding(userLen, l_size).to(self.device).to(torch.float)
        self.ecoEmbed_g = ScaledEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_g = ScaledEmbedding(itemLen, l_size).to(self.device).to(torch.float)

        self.globalBias_d = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.globalBias_d.weight.data += 0.5
        self.globalBias_d.weight.requires_grad = False
        self.ecoBias_d = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.userBias_d = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.itemBias_d = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.userEmbed_d = ScaledEmbedding(userLen, l_size).to(self.device).to(torch.float)
        self.ecoEmbed_d = ScaledEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_d = ScaledEmbedding(itemLen, l_size).to(self.device).to(torch.float)

        self.globalBias_a = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.globalBias_a.weight.requires_grad = False
        self.userBias_a = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.userBias_a.weight.data.uniform_(0.0, 0.05)
        self.itemBias_a = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.ecoBias_a = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.itemBias_a.weight.data.uniform_(0.0, 0.05)
        self.userEmbed_a = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
        self.userEmbed_a.weight.data.uniform_(-0.01, 0.01)
        self.itemEmbed_a = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_a.weight.data.uniform_(-0.01, 0.01)
        self.ecoEmbed_a = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.ecoEmbed_a.weight.data.uniform_(-0.01, 0.01)

        self.globalBias_b = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.globalBias_b.weight.requires_grad = False
        self.userBias_b = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.ecoBias_b = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.userBias_b.weight.data.uniform_(0.0, 0.05)
        self.ecoBias_b.weight.data.uniform_(0.0, 0.05)
        self.itemBias_b = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.itemBias_b.weight.data.uniform_(0.0, 0.05)
        self.userEmbed_b = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
        self.userEmbed_b.weight.data.uniform_(-0.01, 0.01)
        self.itemEmbed_b = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_b.weight.data.uniform_(-0.01, 0.01)
        self.ecoEmbed_b = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_b.weight.data.uniform_(-0.01, 0.01)

        self.ecoEmbed_l = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.globalBias_l = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.globalBias_l.weight.data += 1
        self.globalBias_l.weight.requires_grad = False
        self.userBias_l = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.userBias_l.weight.data.uniform_(0.0, 0.05)
        self.itemBias_l = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.itemBias_l.weight.data.uniform_(0.0, 0.05)
        self.ecoBias_l = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
        self.ecoBias_l.weight.data.uniform_(0.0, 0.05)
        self.userEmbed_l = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
        self.userEmbed_l.weight.data.uniform_(-0.01, 0.01)
        self.itemEmbed_l = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.itemEmbed_l.weight.data.uniform_(-0.01, 0.01)
        self.ecoEmbed_l = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
        self.ecoEmbed_l.weight.data.uniform_(-0.01, 0.01)

        self.reference_point = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.reference_point.weight.data = torch.ones_like(self.reference_point.weight.data) * 1.5
        #		 self.reference_point.weight.requires_grad=False
        self.to(self.device)
        self.grads = {}
    def ecoForward(self, items):
        ecoBias_a = self.ecoBias_a(items)
        ecoEmbed_a = self.ecoEmbed_a(items)
        itemEmbed_a = self.itemEmbed_a(items)

        alpha = ecoBias_a + torch.mul(ecoEmbed_a, itemEmbed_a).sum(1).view(-1, 1)
        return alpha

    def forward(self, users, items):
        distribution = self.distribution[items].to(self.device)
        reference_point = self.reference_point(users)
        #		 print(users.shape[0],items.shape[0])
        price = self.item_price[items].view(-1, 1).expand(users.shape[0], 5).to(self.device)

        # calculate value
        globalBias_a = self.globalBias_a(torch.tensor(0).to(self.device))
        userBias_a = self.userBias_a(users)
        itemBias_a = self.itemBias_a(items)
        userEmbed_a = self.userEmbed_a(users)
        itemEmbed_a = self.itemEmbed_a(items)

        globalBias_b = self.globalBias_b(torch.tensor(0).to(self.device))
        userBias_b = self.userBias_b(users)
        itemBias_b = self.itemBias_b(items)
        userEmbed_b = self.userEmbed_b(users)
        itemEmbed_b = self.itemEmbed_b(items)

        globalBias_l = self.globalBias_l(torch.tensor(0).to(self.device))
        userBias_l = self.userBias_l(users)
        itemBias_l = self.itemBias_l(items)
        userEmbed_l = self.userEmbed_l(users)
        itemEmbed_l = self.itemEmbed_l(items)

        alpha = globalBias_a + userBias_a + itemBias_a + torch.mul(userEmbed_a, itemEmbed_a).sum(1).view(-1, 1)
        beta = globalBias_b + userBias_b + itemBias_b + torch.mul(userEmbed_b, itemEmbed_b).sum(1).view(-1, 1)
        lamda = globalBias_l + userBias_l + itemBias_l + torch.mul(userEmbed_l, itemEmbed_l).sum(1).view(-1, 1)

        rating = torch.tensor([1., 2., 3., 4., 5.]).expand(users.shape[0], 5).to(self.device)
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
        userBias_g = self.userBias_g(users)
        itemBias_g = self.itemBias_g(items)
        userEmbed_g = self.userEmbed_g(users)
        itemEmbed_g = self.itemEmbed_g(items)

        globalBias_d = self.globalBias_d(torch.tensor(0).to(self.device))
        userBias_d = self.userBias_d(users)
        itemBias_d = self.itemBias_d(items)
        userEmbed_d = self.userEmbed_d(users)
        itemEmbed_d = self.itemEmbed_d(items)

        gamma = globalBias_g + userBias_g + itemBias_g + torch.mul(userEmbed_g, itemEmbed_g).sum(1).view(-1, 1)
        delta = globalBias_d + userBias_d + itemBias_d + torch.mul(userEmbed_d, itemEmbed_d).sum(1).view(-1, 1)

        gamma_ = gamma.expand(users.shape[0], 5)
        delta_ = delta.expand(users.shape[0], 5)
        w_exp = torch.mul(x_binary_pos, gamma_) + torch.mul(x_binary_neg, delta_)

        w_nominator = distribution.pow(w_exp)
        w_denominator = (distribution.pow(w_exp) + (torch.ones_like(distribution).to(self.device) - distribution).pow(
            w_exp)).pow(1 / w_exp)
        weight = torch.div(w_nominator, w_denominator)

        #		 self.userBias_g.weight.register_hook(self.save_grad('userBias_g'))
        #		 self.itemBias_g.weight.register_hook(self.save_grad('itemBias_g'))
        #		 self.userEmbed_g.weight.register_hook(self.save_grad('userEmbed_g'))
        #		 self.itemEmbed_g.weight.register_hook(self.save_grad('itemEmbed_g'))
        return torch.mul(weight, value).sum(1)

    def loss(self, users, items, negItems):
        nusers = users.view(-1, 1).to(self.device)
        nusers = nusers.expand(nusers.shape[0], self.params['negNum_train']).reshape(-1).to(self.device)

        pOut = self.forward(users, items).view(-1,1)  # .expand(users.shape[0], self.params['negNum_train']).reshape(-1, 1)
        nOut = self.forward(nusers, negItems).reshape(-1, self.params['negNum_train'])
        Out = torch.cat((pOut, nOut), dim=1)

            #         print(Out.shape)
            #         print(nOut.shape)
            #         input()
        criterion = nn.LogSoftmax(dim=1)
        res = criterion(Out)[:, 0]
        loss = torch.mean(res)
        neg = data_loader.get_env_neg(items, self.params['negNum_train'])
        for j in data_loader.items:
            if data_loader.items[j][1] == items and data_loader.items[j][0] == 1:
                env = 1
                break
            elif data_loader.items[j][1] == items and data_loader.items[j][0] == 0:
                env = 0
                break
        for n in range(len(neg)):
            neg[n] = neg[n][1]
        if env == 1:
            Out = torch.cat((pOut, self.ecoForward(torch.tensor(neg)).reshape(-1, self.params['negNum_train'])), dim=1)
            res = criterion(Out)[:, 0]
            loss += torch.mean(res)
        else:
            Out = -torch.cat((pOut, self.ecoForward(torch.tensor(neg)).reshape(-1, self.params['negNum_train'])), dim=1)
            res = criterion(Out)[:, 0]
            loss += torch.mean(res)
        return -loss

    def get_grads(self):
        return self.grads

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad

        return hook

"""
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

fig, axes = plt.subplots(3, 2, figsize=(10, 15))

X, y = load_digits(return_X_y=True)

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()
"""

if __name__ == '__main__':
    params = dict()
    params['lr'] = 1e-4
    params['batch_size'] = 1
    params['epoch_limit'] = 50
    params['w_decay'] = 5e-4
    params['negNum_test'] = 52
    params['epsilon'] = 1e-4
    params['negNum_train'] = 2
    params['l_size'] = 128
    params['train_device'] = 'cpu'
    params['test_device'] = 'cpu'
    params['lambda'] = 1.
    params['test_per_train'] = 5
    itemprice = np.load(r"C:\\Users\march\Risk-Aware-Recommnedation-Model\data\Movielens1M_item_price.npy")
    category1 = 'newTrainSamples'
    category2 = 'newTestSamples'
    catAll = 'AllSamples'
    metaCat = 'Appliances'

    train, test = data_loader.read_data(category1, category2)
    userNum, itemNum = data_loader.get_datasize(catAll)
    data_loader.get_ecoScores('Appliances', catAll)
    AllSamples = data_loader.read_AllSamples(catAll)
    frequency = data_loader.get_itemDist(AllSamples, itemNum)
    frequency = data_loader.approx_Gaussian(frequency)

    trainset = data_loader.TransactionData(train, userNum, itemNum, frequency)
    trainLoader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=False, num_workers=0)

    testset = data_loader.UserTransactionData(test, userNum, itemNum, trainset.userHist)
    testset.set_negN(params['negNum_test'])
    testLoader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    model = PT(userLen=userNum, itemLen=itemNum, distribution=frequency, params=params, item_price=itemprice)
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
        # print(model.state_dict())
        # fig, axes = plt.subplots(3, 2, figsize=(10, 15))

        # X, y = load_digits(return_X_y=True)

        # title = "Learning Curves (Naive Bayes)"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

        # estimator = GaussianNB()
        # plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
        #                   cv=cv, n_jobs=4)

        if epoch % params['test_per_train'] == 0:
            print('starting test...')
            model.device = params['test_device']
            model.to(model.device)
            L = len(testLoader.dataset)
            pbar = tqdm(total=L)
            with torch.no_grad():
                scoreDict = dict()
                ecoDict = dict()
                for i, batchData in enumerate(testLoader):
                    user = torch.LongTensor(batchData['user']).to(model.device)
                    posItems = torch.LongTensor(batchData['posItem']).to(model.device)
                    negItems = torch.LongTensor(batchData['negItem']).to(model.device)

                    items = torch.cat((posItems, negItems), 1).view(-1)
                    users = user.expand(items.shape[0])

                    score = model.forward(users, items)
                    #ecoScore = model.ecoForward(items)
                    scoreHeap = list()
                    ecoHeap = list()
                    for j in range(score.shape[0]):
                        gt = False
                        et = False
                        if j < posItems.shape[1]:
                            gt = True
                        for k in data_loader.items:
                            if (data_loader.items[k][1] == items[j]):
                                if data_loader.items[k][0] == 1:
                                    et = True
                        heappush(scoreHeap, (1 - score[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
                        heappush(ecoHeap, (1 - score[j].cpu().numpy(), (0 + items[j].cpu().numpy(), et)))
                    scores = list()
                    ecoScores = list()
                    candidate = len(scoreHeap)
                    for k in range(candidate):
                        scores.append(heappop(scoreHeap))
                        ecoScores.append(heappop(ecoHeap))
                    pbar.update(1)
                    scoreDict[user[0]] = (scores, posItems.shape[1])
                    ecoDict[user[0]] = (ecoScores, posItems.shape[1])
            pbar.close()
            testResult = evaluation.ranking_performance(scoreDict, params['negNum_test'])
            print("ECO-RESULTS")
            testResult = evaluation.ranking_performance(ecoDict, params['negNum_test'])

