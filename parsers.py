#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

class Parser(object):

   
    def str_to_bool(text):
        if text.lower() == "true":
            return True
        elif text.lower() == "false":
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean Value Expected')

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--path", default="C:\\Users\\Toshiba\\Desktop\\my_project", help="Base Path for the Folder")
        parser.add_argument("--project", default="my_project", help="Project Folder")
        parser.add_argument("--folder_suffix", default="", help="Folder Name Suffix")
        parser.add_argument("--dataset", default="D1", help="Name of the Dataset")
        parser.add_argument("--opt", default="custom", help="Optimizer : custom, adam, rmsprop, sgd, momentum.")
        parser.add_argument("--lr", default=0.0001, help="Learning Rate", type=float)
        parser.add_argument("--hidden", default=512, help="Number of Neurons in the Hidden Layer", type=int)
        parser.add_argument("--latent_embedding_dim", default=51, help="Number of Neurons in the Embedding", type=int)
        parser.add_argument("--lagrange", default=0.5, help="Lagrange Constant", type=float)
        parser.add_argument("--alpha", default=2, help="Alpha", type=float)
        parser.add_argument("--batch_size", default=500, help="Batch Size", type=int)
        parser.add_argument("--dropout", default=, help="Dropout Probab. for Pre-Final Layer", type=float)
        parser.add_argument("--top_k", default=44, help="Number of selected antenna", type=int)
        parser.add_argument("--max_epochs", default=150, help="Maximum Number of Epochs", type=int)
        parser.add_argument("--patience", default=2, help="Patience", type=int)
        parser.add_argument("--patience_increase", default=2, help="Patience Increase", type=int)
        parser.add_argument("--improvement_threshold", default=2, help="Improvement Threshold for Patience", type=int)
        parser.add_argument("--save_after", default=0, help="Save after how many Epochs?", type=int)
        parser.add_argument("--epoch_test_freq", default=1, help="test Epoch Frequency when training", type=int)
        parser.add_argument("--have_patience", default=False, help="Patience is virtue. NOT!", type=Parser.str_to_bool)
        parser.add_argument("--retrain", default=False, type=Parser.str_to_bool, help="Retrain Flag")
        parser.add_argument("--load", default=True, type=Parser.str_to_bool, help="Load Model to calculate accuracy")
        self.parser = parser

    def get_parser(self):
        return self.parser

args, _ = Parser().get_parser().parse_known_args()
print("Base Path:", args.path)
print("Project Folder:", args.project)
print(args)

