'''
author: Aditi Anant Munjekar
'''

import numpy as np
from scipy.stats import norm

def condprob(x, n, y, params):
    return norm.pdf(x, params[n][y]['mean'], params[n][y]['var'])

def learn(data, N, Y):
    params = {}
    for n in range(N):
        params[n] = {}
        for y in range(Y):
            params[n][y] = {}
            subset = []
            for obs in data:
                if obs[-1] == y:
                    subset.append(obs[n])
            params[n][y]['mean'] = np.mean(subset)
            params[n][y]['var'] = np.var(subset)

    return params

def classify(obs, params, N, Y):
    ans = []
    for y in range(Y):
        prob = 1
        for n in range(N):
            prob *= condprob(obs[n], n, y, params)
        ans.append(prob)
    return ans

def majority(data):
    count, count0, count1 = 0, 0, 0
    for i in (data):
        if i[-1] == 1.0:
            count1 = count + 1
        if i[-1] == 0.0:
            count0 += 1
    count = max(count0, count1)
    print(count)
    majority = (count / len(data))*100
    return majority

def demo():
    #complete data
    data = np.array([[float(x) for x in line.strip().split(",")] for line in open("banknote.train").readlines()])
    print('Loaded %d observations.' % len(data))

    num_of_folds = 3
    fold_size = int(len(data) / num_of_folds)

    #training data for 1st fold
    datatrain1 = data[:fold_size]#training data for 1st fold

    #calculating the N and Y
    N = len(datatrain1[0]) - 1
    distinct = []
    list = []
    for i in range(len(datatrain1)):
        list.append(datatrain1[i][-1])
        for j in list:
            if j not in distinct:
                distinct.append(j)
    Y = len(distinct)

    params = learn(datatrain1, N, Y)

    correct = 0
    for obs in data[fold_size:]:# testing data for 1st fold
        result = classify(obs, params, N, Y)
        result = np.array(result) / np.sum(result)
        if np.argmax(result) == obs[-1]:
            correct += 1
    accuracy1 = (correct / (len(data)-fold_size)) * 100
    print('Accuracy after 1nd fold: %.3f%%' % accuracy1)

    #for training 2nd fold
    datatrain2 = data[fold_size:(len(data) - fold_size)]

    params = learn(datatrain2, N, Y)

    #for testing
    correct = 0
    for obs in data[:fold_size]:#testing data for 2nd fold
        result = classify(obs, params, N, Y)
        result = np.array(result) / np.sum(result)
        if np.argmax(result) == obs[-1]:
            correct += 1
    for obs in data[2*fold_size:]:
        result = classify(obs, params, N, Y)
        result = np.array(result) / np.sum(result)
        if np.argmax(result) == obs[-1]:
            correct += 1

    accuracy2 = (correct/(len(data)-fold_size))*100
    print('Accuracy after 2nd fold: %.3f%%' % accuracy2)

    #for training for 3rd fold
    datatrain3 = data[2*fold_size:]

    params = learn(datatrain3, N, Y)

    #for testing for 3rd fold
    correct = 0
    for obs in data[:2*fold_size]:#testing data 3rd fold
        result = classify(obs, params, N, Y)
        result = np.array(result) / np.sum(result)
        if np.argmax(result) == obs[-1]:
            correct += 1

    accuracy3 = (correct/(len(data)-fold_size))*100
    print('Accuracy after 3nd fold: %.3f%%' % accuracy3)

    print('Mean is: %.3f%%'% ((accuracy1+accuracy2+accuracy3)/3))
    majority_value = majority(data)
    print('Majority baseline of dataset is: %.3f%% '% majority_value)

if __name__ == '__main__':
    demo()



