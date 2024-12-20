#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import utils
import tensorflow as tf
import custom_optimizer
from custom_optimizer import CustomOptimizer

class Config(object):
    def __init__(self, args):
        self.codebase_root_path = args.path
        self.folder_suffix = args.folder_suffix
        self.project_name = args.project
        self.dataset_name = args.dataset
        self.retrain = args.retrain
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.patience = args.patience
        self.patience_increase = args.patience_increase
        self.improvement_threshold = args.improvement_threshold
        self.save_after = args.save_after
        self.epoch_test_freq = args.epoch_test_freq
        self.load = args.load
        self.have_patience = args.have_patience
        
        class Solver(object):
            def __init__(self, t_args):
                self.learning_rate = t_args.lr
                self.hidden_dim = t_args.hidden
                self.latent_embedding_dim = t_args.latent_embedding_dim
                self.dropout = t_args.dropout 
                self.top_k = t_args.top_k
                self.lagrange_const = t_args.lagrange
                self.alpha = t_args.alpha
                if t_args.opt.lower() not in ["custom", "adam", "rmsprop", "sgd", "momentum"]:
                    raise ValueError('Undefined type of optimizer')
                else:  
                    self.optimizer = {"custom": CustomOptimizer,
                                      "adam": tf.train.AdamOptimizer,
                                      "rmsprop": tf.train.RMSPropOptimizer,
                                      "sgd": tf.train.GradientDescentOptimizer,
                                      "momentum": tf.train.MomentumOptimizer}[t_args.opt.lower()]

        self.solver = Solver(args)
        self.project_path, self.project_prefix_path, self.dataset_path, self.train_path, self.test_path,self.validation_path,self.ckptdir_path = self.set_paths()
        self.features_dim, self.labels_dim = self.set_dims()

    def set_paths(self):
        project_path = utils.path_exists(self.codebase_root_path)
        project_prefix_path = "" #utils.path_exists(os.path.join(self.codebase_root_path, self.project_name, self.folder_suffix))
        dataset_path = utils.path_exists(os.path.join(self.codebase_root_path, "venv", self.dataset_name))
        ckptdir_path = utils.path_exists(os.path.join(self.codebase_root_path, "checkpoints"))
        train_path = os.path.join(dataset_path, self.dataset_name + "-train")
        test_path = os.path.join(dataset_path, self.dataset_name + "-test")
        validation_path = os.path.join(dataset_path, self.dataset_name + "-valid")
        
        
        return project_path, project_prefix_path, dataset_path, train_path, test_path, validation_path, ckptdir_path, 
        
         
    def set_dims(self):
        with open(os.path.join(self.dataset_path, "count.txt"), "r") as f:
            return [int(i) for i in f.read().split("\n") if i != ""]

        
    

