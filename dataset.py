#!/usr/bin/env python
# coding: utf-8

# In[21]:


# %load dataset.py
get_ipython().system('pip install liac-arff')
import arff
import numpy as np
import os



class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.train, self.test, self.validation = None, None, None
        self.path = self.config.dataset_path

    def get_data(self, path, noise=False):
        data = np.load(path)
        if noise == True :
            data = data + np.random.normal(0, 0.001, data.shape)
        return data

    def get_train(self):
        if self.train == None:
            X = self.get_data(self.config.train_path + "-features.pkl")
            Y = self.get_data(self.config.train_path + "-labels.pkl")
            X, Y = self.eliminate_data(X, Y)
            self.train = X, Y
        else:
            X, Y = self.train

        return X, Y

    def get_validation(self):
        if self.validation == None:
            X = self.get_data(os.path.join(self.config.validation_path + "-features.pkl"))
            Y = self.get_data(os.path.join(self.config.validation_path + "-labels.pkl"))
            self.validation = X, Y
        else:
            X, Y = self.validation
        return X, Y
   
    def get_test(self):
        if self.test == None:
            X = self.get_data(self.config.test_path + "-features.pkl")
            Y = self.get_data(self.config.test_path + "-labels.pkl")
            self.test = X, Y
        else:
            X, Y = self.test

        return X, Y

    def next_batch(self, data):
        if data.lower() not in ["train", "test", "validation"]:
            raise ValueError
        func = {"train": self.get_train, "test": self.get_test, "validation": self.get_validation}[data.lower()]
        X, Y = func()
        start = 0
        batch_size = self.config.batch_size
        total = len(X)
        batch_num = int(total / batch_size)  # fix the last batch
        while start < total:
            end = min(start + batch_size, total)
            x = X[start:end, :]
            y = Y[start:end, :]

            if end == total:
                x = np.r_[x, X[0:start + batch_size - total, :]]
                y = np.r_[y, Y[0:start + batch_size - total, :]]
            start += batch_size
            yield (x, y, int(batch_num))

    def eliminate_data(self, X, Y):
        num_labels = Y.shape[1]
        full_true = np.ones(num_labels)
        full_false = np.zeros(num_labels)

        i = 0
        while (i < len(Y)):
            if (Y[i] == full_true).all() or (Y[i] == full_false).all():
                Y = np.delete(Y, i, axis=0)
                X = np.delete(X, i, axis=0)
            else:
                i = i + 1
        return X, Y


# In[ ]:




