""" 
This file is a sample code of model.py
"""


import tensorflow as tf


"""
model_configs : all the arguments necessary for your model design.

EX) model_configs = {"num_blocks" : 6, "activation_func" : 'relu', "norm_layer" : 'batch_norm'} 
"""
model_configs = {} # fill in your model configs


""" You can change the model name and implement your model. """
class Classifier(tf.keras.Model):
    def __init__(self, num_classes=None, **kwargs):
        super().__init__()
    
    def call(self, x):
        return self.fc(x)
    


""" [IMPORTANT]
get_classifier function will be imported in evaluation file.
You should pass all the model configuration arguments in the get_classifier function 
so that we can successfully load your exact model
saved in the submitted model checktpoint file.
"""
def get_classifier(num_classes=None):
    return Classifier(num_classes=num_classes, **model_configs)




