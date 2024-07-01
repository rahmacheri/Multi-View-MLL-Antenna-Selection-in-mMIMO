#!/usr/bin/env python
# coding: utf-8

# In[4]:


#@Time      :2019/3/28 16:28
#@Author    :zhounan
# @FileName: model.py
import os
import sys
import time
import utils
import numpy as np
from scipy.io import savemat
import tensorflow as tf
from parsers import Parser
from config import Config
from network import Network
from dataset import DataSet
from eval_performance import eval_performance

class Model(object):
    def __init__(self, config):
        self.epoch_count = 0
        self.config = config
        self.data = DataSet(config)
        self.add_placeholders()
        self.summarizer = tf.summary
        self.net = Network(config, self.summarizer)
        self.optimizer = self.config.solver.optimizer
        self.pred = self.net.prediction(self.x, self.keep_prob)
        self.loss = self.net.loss(self.x, self.y, self.keep_prob)
        self.evaluate= eval_performance(config)
        self.summarizer.scalar("loss", self.loss)
        self.train_op = self.net.train_step(self.loss, self.config.solver.optimizer, self.learning_rate)
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
        self.local_init = tf.local_variables_initializer()
        
       

    def add_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.features_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.labels_dim])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.top_k = tf.placeholder(tf.float32)
     
    def save_model(self, sess, checkpoint_path):
        # Create the checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Save the model
        self.saver.save(sess, os.path.join(checkpoint_path, "model.ckpt"))

    def run_epoch(self, sess, data_type, summary_writer, epoch, lr):
        err = list()
        i = 0
        merged_summary = self.summarizer.merge_all()

        for X, Y, tot in self.data.next_batch(data_type):

            # Wx1 = self.net.Wx1
            # Wx1_val = sess.run(Wx1)
            # grads = sess.run(tf.gradients(self.loss, [Wx1]), )
            # X, Y = self.data.get_validation()
            # feed_dict = {self.x: X,
            #              self.y: Y,
            #              self.keep_prob: 1}
            # Fx = self.net.Fx(self.x, self.keep_prob)
            # Fe = self.net.Fe(self.y, self.keep_prob)
            # pre_loss = self.net.loss_prediction(self.y, self.keep_prob)
            #
            # em_loss = self.net.embedding_loss(Fx, Fe)
            # output_loss = self.net.output_loss(pre_loss, self.y)
            # em_loss, output_loss = sess.run([em_loss, output_loss], feed_dict=feed_dict)
            #
            # Fx, Fe, pre_loss = sess.run([Fx, Fe, pre_loss], feed_dict=feed_dict)
            # print('val emloss ', em_loss)
            # print('val output_loss', output_loss * 5 / X.shape[0])
            feed_dict = {self.x: X,
                         self.y: Y,
                         self.keep_prob: self.config.solver.dropout,
                         self.learning_rate: lr}
            # Wx1 = self.net.Wx1
            # Wx1_val = sess.run(Wx1)
            # grads = sess.run(tf.gradients(self.loss, [Wx1]), feed_dict=feed_dict)[0]
            # print(grads)
            if not self.config.load:
                _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                mean_loss = loss / X.shape[0]

                # write train log
                output = "Epoch ({}) Batch({}) - Mean Loss : {}".format(self.epoch_count, i, mean_loss)
                with open("{}_train.log".format(self.config.project_prefix_path), "a+") as log:
                    log.write(output + "\n")
                print("   {}".format(output), end='\r')

        X, Y = self.data.get_train()
        feed_dict = {self.x: X,
                     self.y: Y,
                     self.keep_prob: self.config.solver.dropout}
        summary, loss = sess.run([merged_summary, self.loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary, epoch)
        mean_loss = loss / X.shape[0]
        return mean_loss

    def run_eval(self, sess, data_type, epoch, summary_writer=None):
        merged_summary = self.summarizer.merge_all()
        i = 0
        if data_type == "validation":
            X, Y = self.data.get_validation()
        elif data_type == 'test':
            X, Y = self.data.get_test()
        feed_dict = {self.x: X,
                     self.y: Y,
                     self.keep_prob: 1}

        # if only test data summary_writer is None
        if summary_writer is not None:
            summ, loss = sess.run([merged_summary, self.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summ, epoch)

        loss, Y_pred = sess.run([self.loss, self.pred], feed_dict=feed_dict)
        metrics = self.evaluate.evaluate(predictions=Y_pred, labels=Y)

        mean_loss = loss / X.shape[0]
        return mean_loss, metrics
    
    def get_prediction(self, sess):
        top_k= self.config.solver.top_k
        X, Y = self.data.get_test()
        feed_dict = {self.x: X,
                 self.y: Y,
                 self.keep_prob: 1}
        start_time = time.time() 
        _, predictions = sess.run([self.loss, self.pred], feed_dict=feed_dict)
        end_time = time.time() # Measure the end time
        inference_Time = end_time - start_time  # Calculate inference time      
   # Sort predictions for each sample in descending order
        sorted_indices = np.argsort(-predictions, axis=1)
    # Initialize selected labels matrix
        selected_labels = np.zeros_like(predictions)
    # Set the top k labels with the highest probabilities to 1
        for i in range(len(selected_labels)):
            top_indices = sorted_indices[i, :top_k]
            selected_labels[i, top_indices] = 1

        return selected_labels,inference_Time 
    
    def evall(self,sess,summary_writers):
        
        predlabels,inference_Time = self.get_prediction(sess)
        inference_time = inference_Time/ 2000
        test_loss, test_metrics = self.run_eval(sess, "test",  summary_writers['test'])       
        returnDict = {"test_loss": test_loss, 'test_metrics': test_metrics}
        return returnDict , predlabels, inference_time
        
    
       
    def init_summaries(self, sess):

        path_ = "../bin/results/tensorboard"
        summary_writer_train = tf.summary.FileWriter(path_ + "/train", sess.graph)
        summary_writer_val = tf.summary.FileWriter(path_ + "/val", sess.graph)
        summary_writer_test = tf.summary.FileWriter(path_ + "/test", sess.graph)
        summary_writers = {'train': summary_writer_train, 'val': summary_writer_val, 'test': summary_writer_test}
        return summary_writers

    def fit(self, sess, summary_writers):
        '''
         - Patience Method :
         + Train for particular no. of epochs, and based on the frequency, evaluate the model using validation data.
         + If Validation Loss increases, decrease the patience counter.
         + If patience becomes less than a certain threshold, devide learning rate by 10 and switch back to old model
         + If learning rate is lesser than a certain
        '''
        sess.run(self.init)
        max_epochs = self.config.max_epochs
        patience = self.config.patience
        patience_increase = self.config.patience_increase
        improvement_threshold = self.config.improvement_threshold
        best_validation_loss = 1e6
        self.epoch_count = 0
        best_step, learning_rate = -1, self.config.solver.learning_rate

        while self.epoch_count < max_epochs:

            # Xval, Yval = self.data.get_validation()
            # feed_dict = {self.x: Xval,
            #              self.y: Yval,
            #              self.keep_prob: 1}
            # Fx = self.net.Fx(self.x, self.keep_prob)
            # Fe = self.net.Fe(self.y, self.keep_prob)
            # pre_loss = self.net.loss_prediction(self.y, self.keep_prob)
            #
            # em_loss = self.net.embedding_loss(Fx, Fe)
            # output_loss = self.net.output_loss(pre_loss, self.y)
            # em_loss, output_loss = sess.run([em_loss, output_loss], feed_dict=feed_dict)
            #
            # Fx, Fe, pre_loss = sess.run([Fx, Fe, pre_loss], feed_dict=feed_dict)
            # print('val emloss ', em_loss)
            # print('val output_loss', output_loss * 5 / Xval.shape[0])

            learning_rate_epoch = learning_rate * np.power(0.98, self.epoch_count)
            print('learning_rate: {}'.format(learning_rate_epoch))
            train_loss = self.run_epoch(sess, "train", summary_writers['train'], self.epoch_count, learning_rate_epoch)

            if self.epoch_count % self.config.epoch_test_freq == 0:

                val_loss, val_metrics = self.run_eval(sess, "validation", self.epoch_count, summary_writers['val'])

                output = "=> time:{}, epochs{}/{},  Training : Loss = {:.2f} | Validation : Loss = {:.2f}"                     .format(time.strftime('%H:%M:%S', time.localtime(time.time())), self.epoch_count, max_epochs,
                            train_loss, val_loss)

                # write test log
                with open("{}validation.log".format(self.config.project_prefix_path), "a+") as f:
                    output_ = output + "\n=> val: Coverage = {}, Average Precision = {}, Micro Precision = {}, Micro Recall = {}, Micro F Score = {}"                         .format(val_metrics['coverage'], val_metrics['average_precision'],
                                val_metrics['micro_precision'], val_metrics['micro_recall'], val_metrics['micro_f1'])
                    output_ += "\n=> val : Hamming Loss = {}, Macro Precision = {}, Macro Recall = {}, Macro F Score = {}\n\n"                         .format(val_metrics['hamming_loss'], val_metrics['macro_precision'],
                                val_metrics['macro_recall'], val_metrics['macro_f1'])
                    f.write(output_)
                print(output)
                print()

                if self.config.have_patience:
                    if val_loss < best_validation_loss:
                        if val_loss < best_validation_loss * improvement_threshold:
                            self.saver.save(sess, self.config.ckptdir_path + "model_best.ckpt")
                            best_validation_loss = val_loss
                            best_step = self.epoch_count
                    else:
                        if patience < 1:
                            self.saver.restore(sess, self.config.ckptdir_path + "model_best.ckpt")
                            if learning_rate <= 0.00001:
                                print("=> Breaking by Patience Method")
                                break
                            else:
                                learning_rate *= 0.8
                                patience = self.config.patience

                                print("\033[91m=> Learning rate dropped to {}\033[0m".format(learning_rate))
                        else:
                            patience -= 1

            self.epoch_count += 1
       

                  
                    
        print("=> Best epoch : {}".format(best_step))

        returnDict = {"val_loss": val_loss, 'val_metrics': val_metrics}
        returnDict["train_loss"] = train_loss
        return returnDict

