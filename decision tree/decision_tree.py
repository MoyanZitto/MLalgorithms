# -*- coding: utf-8 -*-
"""
Created on Mon May 30 01:09:58 2016

@author: Moyan

We will use decision tree(ID3 and C4.5) to predict the contact lens type(Machine
 Learning in action, chp3)
 
The data structure will be kd-tree

    the acc of this algorithm(ID3) :
    the acc of this algorithm(C4.5):
    the acc of sklearn(ID3)        :
    the acc of sklearn(C4.5)       :
"""
import numpy as np


class DecisionTree(object):
    def __init__(self, data_path, algorithm='C4.5'):
        self.data_path = data_path
        self.algorithm = algorithm
        
        
        
    def make_data(self):
        code = [{'young':0,'pre':1,'presbyopic':2},
                {'myope':0,'hyper':1},
                {'no':0,'yes':1},
                {'reduced':0,'normal':1},
                {'no lenses':0, 'soft':1, 'hard':2}]
                
        self.data = []
        with open(self.data_path) as f:
            for line in f:
                self.data.append([code[index][word] for (index,word) in enumerate(line.strip().split('\t'))])
        
        self.data = np.array(self.data)
        self.data = np.random.shuffle(self.data)
    
    def get_fold_data(self):
        # use 4-fold validation to evaluate the performance
        split = self.data.shape[0]/5
        for i in range(4):
            test = self.data[i*split:(i+1)*split]
            train = np.array(list(set(self.data) - set(Y)))
            X_train = train[:,:-1]
            Y_train = train[:,-1]
            X_test = test[:,:-1]
            Y_test = test[:,-1]
            yield (X_train, X_test, Y_train, Y_test)
            
    
    def build_tree(self, X_train, X_test):
        
    
    def prune_tree(self):
        pass

    def predict(self):
        pass
    
    def show_proformance(self):
        pass
    
    def run(self):
        pass
        
        
        
if __name__=='__main__':
    tree = DecisionTree('data.txt')
    tree.make_data()
        
        
        
        
        
        
        
        
        
        