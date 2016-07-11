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
import copy

class Node(object):
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None
        self.father = None
        self.cut_dim = None

class kd_tree(object):
    def __init__(self,path):
        self.path = path
        self.train_data, self.test_data = self.make_data(0.5)
        self.root = self.build_kdtree(self.train_data)
    
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
        data[:,:3] = self.normalization(data[:,:3])
        np.random.shuffle(data)
        split_point = int(data.shape[0]*ratio)
        data_train = data[:split_point]
        data_test = data[split_point:]
        return data_train, data_test

    def build_kdtree(self, data, cut_dim=0, father=None):
        if len(data)==0:
            return None
        new_node = Node()
        new_node.father = father
        new_node.cut_dim = cut_dim
        data = sorted(data, key=lambda d:d[cut_dim])
        new_node.data = data[len(data)/2]
        new_node.left = self.build_kdtree(data[:len(data)/2], (cut_dim+1)%3, father=new_node)
        new_node.right = self.build_kdtree(data[len(data)/2+1:],(cut_dim+1)%3, father = new_node)
        return new_node
    
    def euler(self, x1,x2):
        return np.dot((x1[:3]-x2[:3]),(x1[:3]-x2[:3]))

    
    def search_branch(self, x, root, neardis, nearest): 
        if root==None:
            return nearest, neardis
            
        cur = root       
        cur_dis = self.euler(x,cur.data)
        if cur_dis < neardis:
            nearest = cur.data
            neardis = cur_dis
        
        if cur.left==None and cur.right==None:
            cur_dis = self.euler(x,cur.data)
            if cur_dis < neardis:
                nearest = cur.data
                neardis = cur_dis
            return nearest, neardis
            

        
        #如果分割点于x在轴cut_dim上的距离小于neardis，说明分割点两边的子树都在范围内，都应该搜索
        elif np.abs(cur.data[cur.cut_dim]-x[cur.cut_dim])<neardis:        
                nearest, neardis = self.search_branch(x, cur.left, neardis, nearest)
                nearest, neardis = self.search_branch(x, cur.right, neardis, nearest)
        #否则，分割点只有一侧于x相交，【left,cut_dim,right】
        else:
            if x[cur.cut_dim] - cur.data[cur.cut_dim]>0:
                nearest, neardis = self.search_branch(x, cur.right, neardis, nearest)
            else:
                nearest, neardis = self.search_branch(x, cur.left, neardis, nearest)
        
        return nearest, neardis
    
    
    def search(self,x):
        #first find the subspace that x belongs to
        cur = self.root
        while 1:
            if cur.left==None and cur.right==None:
                break
            
            elif cur.left==None:
                cur = cur.right
                
            elif cur.right ==None:
                cur = cur.left
                
            elif x[cur.cut_dim]<cur.data[cur.cut_dim]:
                cur = cur.left

            else:
                cur = cur.right


        assert cur.left==None and cur.right==None

        
        # cur will be the a leaf when this loop end
        nearest = cur.data
        neardis = self.euler(x,nearest)
        
        while cur.father!=None:
            cur_dis = self.euler(x,cur.father.data)
            if cur_dis < neardis:
                nearest = cur.father.data
                neardis = cur_dis

            if np.abs(cur.father.data[cur.father.cut_dim] - x[cur.father.cut_dim])<neardis:
                if cur == cur.father.left:
                    if cur.father.right!=None:
                        cur_data, cur_dis= self.search_branch(x,cur.father.right,neardis, nearest )
                elif cur == cur.father.right:
                    if cur.father.left!=None:
                        cur_data, cur_dis = self.search_branch(x,cur.father.left,neardis, nearest)
            
                if cur_dis < neardis:
                    neardis = cur_dis
                    nearest = cur_data
                
            cur = cur.father

        return nearest
    
     
    
    
    def show(self):
        print self.root.data
        print self.root.left.data
        print self.root.right.data
        
        


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
            
            return p#Counter(topN_cls).most_common(1)[0][0]
            
            
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
    kd_tree = kd_tree('data.txt')
    x = np.array([1.53,1.02,0.67,1])
    train_data = kd_tree.train_data
    min_dis = 1e8

    for data in train_data:
        cur_dis = kd_tree.euler(x,data)
        if cur_dis<min_dis:
            min_dis = cur_dis
            min_data = data

    
    print min_data,min_dis
    kd_min = kd_tree.search(x)
    print kd_min
    print kd_tree.euler(kd_min,x)