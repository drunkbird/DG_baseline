import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model.predict_mothed import lgb_predict,bayes_predict
from model.reduce_dimension_mothed import pca_reduce

if __name__ == '__main__':
    test_x = pd.read_csv('../data/encode_data/local_test_low_encoded.csv')
    # test_x = pca(test_x,'train',0.95)
    train_x = pd.read_csv('../data/encode_data/local_train_low_encoded.csv')
    train_y = train_x.pop('class')
    res = bayes_predict(train_x,train_y,test_x)
    res.to_csv('../data/submission/sub_2.csv',index=False)
