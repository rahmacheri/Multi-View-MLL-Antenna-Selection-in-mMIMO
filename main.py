

import sys
sys.path.append(r'C:\link\to\your\scripts')
import os
import sys
import time
import utils
import numpy as np
from model import Model
import tensorflow as tf
from parsers import Parser
from config import Config
from network import Network
from dataset import DataSet
from eval_performance import eval_performance
from scipy.io import savemat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def init_model(config):
    tf.reset_default_graph()
    tf.set_random_seed(0)
    with tf.variable_scope('Model', reuse=None) as scope:
        model = Model(config)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    if config.load== True:
        print("=> Loading model from checkpoint")
        
        
    # Restore the model
        model.saver.restore(sess, os.path.join(config.ckptdir_path, "model.ckpt"))
        return model, sess 
        

       
    else:
        print("=> No model loaded from checkpoint")
    return model, sess

def train(model, sess):
    with sess:
        
        summary_writers = model.init_summaries(sess)
        loss_dict = model.fit(sess, summary_writers)

         # Save the model after training
        model.save_model(sess, config.ckptdir_path) 
         
    return loss_dict

def test(model, sess):
    with sess:
        summary_writers = model.init_summaries(sess)
        loss_dict = model.evall(sess,summary_writers)
        
    return loss_dict
    

if __name__ == '__main__' :
    args, _ = Parser().get_parser().parse_known_args()
    config = Config(args)
    model, sess = init_model(config)

    if config.load == True:
        print("\033[92m=>\033[0m Testing Model")
        loss_dict,predlabels,inference_time = test(model, sess) 
        test_metrics = loss_dict['test_metrics']
        output = "=> Test Loss : {}".format(loss_dict["test_loss"])
        
        savemat("{}labels.mat".format(config.project_prefix_path), {'predlabels': predlabels})
        
        time = "Elapsed time for predicting labels on test data: {} seconds".format(inference_time)
        with open("{}time.log".format(config.project_prefix_path), "a+") as f:
            f.write(time)

        # Calculate the number of predicted labels (ones) in each row
        num_predicted_labels = np.sum(predlabels, axis=1)

        # Write the number of predicted labels for each row to the log file
        with open("{}num_predicted_labels.log".format(config.project_prefix_path), "a+") as f:
             for num_labels in num_predicted_labels:
                f.write(str(num_labels) + "\n")  
             
        output += "\n=> Test : Coverage = {}, Average Precision = {}".format(test_metrics['coverage'], test_metrics['average_precision'])
        output += "\n=> Test :  Macro Precision = {}".format(test_metrics['macro_precision'])
        output += "\n=>Test :Micro F Score = {}".format(test_metrics['micro_f1'])
        output += "\n=>Test :Macro F Score = {}".format(test_metrics['macro_f1'])
        output += "\n=>Hamming loss = {}".format(test_metrics['hamming_loss'])
    

        with open("{}test_log.log".format(config.project_prefix_path), "a+") as f:
            f.write(output)    
        print("\033[1m\033[92m{}\033[0m\033[0m".format(output))    
        
        
    else:
        print("\033[92m=>\033[0m Training Model")
        loss_dict = train(model, sess)
        val_metrics = loss_dict['val_metrics']
        output = "=> Best Train Loss : {}, valid Loss : {}".format(loss_dict["train_loss"], loss_dict["val_loss"])
        
    
   
    print("\033[1m\033[92m{}\033[0m\033[0m".format(output))

