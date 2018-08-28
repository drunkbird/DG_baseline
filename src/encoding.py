import pandas as pd
import numpy as np
from model.encode import Encode

FEATURES='article,word_seg'.split(',')

def get_encoded_data(name,data_type,df):
    if data_type == 'train':
        label = df.pop('class')
    #编码函数
    data=Encode.cvEncoding(df,FEATURES)
    if data_type == 'train' : data=data.join(label)
    data.to_csv('../data/encode_data/{0}_encoded.csv'.format(name),index=False)
    print('{0}_encoded saving complete'.format(name))

if __name__ == '__main__' :
    print('start encoding')
    df = pd.read_csv('../data/data_low_dimension/train_set_1_low.csv')
    for i in range(2,20):
        df1 = pd.read_csv('../data/data_low_dimension/train_set_%d_low.csv'%i)
        df = df.append(df1)
    print(df)
    df.to_csv('../data/data_low_dimension/local_train_set_low.csv',index=False,header=True)
    df = pd.read_csv('../data/data_low_dimension/local_train_set_low.csv')
    get_encoded_data('local_train_low','train',df)
    # df = pd.read_csv('../data/data_low_dimension/test_set_low.csv')
    # get_encoded_data('local_test_low','test',df)