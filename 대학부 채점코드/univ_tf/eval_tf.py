import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
import os
from model import get_classifier
from sklearn.metrics import f1_score, classification_report


def run_eval(model, loader):    
    y_prob = model.predict(loader)
    y_pred = np.argmax(y_prob, axis=1)
    y_test = np.concatenate([y for x, y in loader], axis=0)
    result = classification_report(y_test, y_pred, output_dict=True)
    
    return pd.DataFrame(result)


def check_across_seeds(accs, f1s, result_df, num_classes=20):
    accs = np.array(accs)
    f1s = np.array(f1s)
    
    assert np.all(np.abs(accs[1:] - accs[:1]) < 1e-1) and np.all(np.abs(f1s[1:] - f1s[:1]) < 1e-1), "test results are not compatible \n{}\n{}".format(accs, f1s)

    print("*** CLASSWISE RESULT ***")
    cwise_result = result_df.loc[['f1-score', 'recall'], [str(i) for i in range(num_classes)]]
    cwise_result = cwise_result.rename(index={'f1-score' : 'f1', 'recall' : 'acc'})
    print(cwise_result)
    
    print("\n*** AVG RESULT ***")
    avg_result = pd.Series({'f1' : result_df.loc['f1-score', 'macro avg'], 'acc' : result_df['accuracy'].values[0]})
    print(avg_result)
    
    
def main():
    ''' 
    Fill in the root directory path into DATA_DIR.
    You must write the subset directory for the specific split (train or valid).
    Under the root directory, the child folders should be "L2_3", "L2_10", ... , "L2_52"
    
    EX) if you named your valid dataset folder as "~/valid"
    then the child directory should be "~/valid/L2_3", "~/valid/L2_10", ... , "~/valid/L2_52"
    
    so you have to write as
    DATA_DIR = "~/valid"
    '''
    DATA_DIR = "YOUR DATA DIRECTORY"           
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_VAR = (0.229 ** 2, 0.224 ** 2, 0.225 ** 2)
    
    
    ''' 
    You need to implement "get_classifier" function that returns your implemented model.
    "get_classifier" should return your model defined with your model configuration.
    Also, you should save your model.
    
    EX)
    model.save_weights('model.ckpt')
    or
    model.save('model.h5')
    '''
    CLF = get_classifier(num_classes=20) 
    CKPT_PATH = "YOUR CHECKPOINT PATH"  
    
    """ if saved with model.save_weights() """
    CLF.load_weights(CKPT_PATH).expect_partial()
    # if necessary
    # CLF.build(input_shape=[None, 224, 224, 3])
    # print(CLF.summary())
    
    """ if saved with model.save() """
    # CLF = load_model(CKPT_PATH)

    
    SEEDS = [0, 5, 10]
    ACC_LIST = []
    F1_LIST = []
    for seed in SEEDS:
        tf.random.set_seed(seed)
        
        loader = tf.keras.preprocessing.image_dataset_from_directory(
            directory=DATA_DIR,
            image_size=(256, 256),
            batch_size=128,
            )
        ## Preprocessing
        augmentation_layer = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.CenterCrop(224,224),
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255),            
            tf.keras.layers.experimental.preprocessing.Normalization(mean=IMAGENET_DEFAULT_MEAN, variance=IMAGENET_DEFAULT_VAR)    
        ])
        loader = loader.map(lambda x,y: (augmentation_layer(x),y))
        
        #gc.collect()
        RESULT_DF = run_eval(CLF, loader)
        ACC_LIST.append(RESULT_DF['accuracy'].values[0])
        F1_LIST.append(RESULT_DF.loc['f1-score', 'macro avg'])

    check_across_seeds(ACC_LIST, F1_LIST, RESULT_DF)
    
if __name__=="__main__":
    main()    
    
 