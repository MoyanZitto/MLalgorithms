# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:55:38 2016

@author: BigMoyan

KNN algorithm for Helen's favorite (Machine Learning in Action, Chap2)

The acc over 30 test cases:

    this algorithm: 0.96
    skleanr knn   : 0.96

"""
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

class KNN(object):
    def __init__(self, path, N, mode='num', split_ratio=0.7):
        self.path = path
        self.split_ratio = split_ratio
        self.N = N
        self.mode=mode
        self.train_data, self.test_data = self.make_data(self.split_ratio)

    def read_data(self,path):
        with open(path,'r') as f:
            data = []
            for line in f:
                data.append([float(i) for i in line.split('\t')])
            return np.array(data)
            
    def normalization(self,data):
        mean = np.mean(data,axis=0)
        std = np.std(data, axis=0)
        data -= mean
        data /= std
        return data
    
    def make_data(self,ratio):
        data = self.read_data(self.path)
        np.random.shuffle(data)
        split_point = int(data.shape[0]*ratio)
        data_train = data[:split_point]
        data_test = data[split_point:]
        X_train = data_train[:,:-1]
        Y_train = data_train[:,-1]
        X_test = data_test[:,:-1]
        Y_test = data_test[:,-1]
        X_train = self.normalization(X_train)
        X_test = self.normalization(X_test)
        return (X_train, Y_train), (X_test, Y_test)
    
    def compute_dis(self,x1,x2):
        return np.dot(x1-x2,x1-x2)
    
    
    def predict(self,x):
        if self.mode=='num':
            assert(isinstance(self.N,int))
            assert(x.shape == self.train_data[0][0].shape)
            diss = []
            for p in self.train_data[0]:
                diss.append(self.compute_dis(x,p))
            
            topN_cls = []
            for i in range(self.N):
                index = np.argmin(diss)
                topN_cls.append(self.train_data[1][index])
                diss[index] = 1e9
            
            return Counter(topN_cls).most_common(1)[0][0]
            
            
        elif self.mode=='dis':
            clss = []
            for i,p in enumerate(self.train_data[0]):
                dis = self.compute_dis(x,p)
                if dis<self.N:
                    clss.append(self.test_data[1][i])
            
            return Counter(clss).most_common(1)[0][0]
            
        else:
            raise ValueError("mode must be 'num' or 'dis'!")

        
    def show_performance(self, algorithm='moyan'):
        if algorithm=='moyan':
            num = self.test_data[0].shape[0]
            count = 0
            for i in range(num):
                if self.predict(self.test_data[0][i])==self.test_data[1][i]:
                    count +=1
            return count/float(num)
        
        elif algorithm=='sklearn':
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(self.train_data[0],self.train_data[1])
            result = knn.predict(self.test_data[0])
            count = np.sum(result==self.test_data[1])
            return count/float(len(self.test_data[1]))
        
    def run(self):
        print "the acc over 300 test data is(by my algorithm):"
        print self.show_performance('moyan')
        print "the acc over 300 test data is(by sklearn algorithm):"
        print self.show_performance('sklearn')


if __name__ == '__main__':
    knn = KNN(path='data.txt', N=5)
    knn.run()
    