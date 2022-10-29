""" 
This file is a sample code of model.py
"""


import torch
import torch.nn as nn


"""
model_configs : all the arguments necessary for your model design.

EX) model_configs = {"num_blocks" : 6, "activation_func" : 'relu', "norm_layer" : 'batch_norm'} 
"""
model_configs = {} # fill in your model configs


""" You can change the model name and implement your model. """
class Classifier(nn.Module):
    def __init__(self, num_classes=20, **kwargs):
        super().__init__()

    def forward(self, x):
        
        return logit
        


""" [IMPORTANT]
get_classifier function will be imported in evaluation file.
You should pass all the model configuration arguments in the get_classifier function 
so that we can successfully load your exact model
saved in the submitted model checktpoint file.
"""
def get_classifier(num_classes=25, **kwargs):
    return Classifier(num_claseses=num_classes, **kwargs)