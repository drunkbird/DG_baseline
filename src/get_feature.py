import pandas as pd
import numpy as np
from model.get_feature_method import *




FEATURE = ['article','word_seg']
if __name__ == '__main__':

    # for fea in FEATURE:
    #     for i in range(1,20):
    #         get_classes_feature('train_set_%d' % i,fea)
    # for fea in FEATURE:
    #     for i in range(1,20):
    #         calcu_tf_idf('train_set', fea, i)
    #         print('%s_train_set_%d_completed'%( fea, i))
    


    for fea in FEATURE:
        df = pd.read_csv('../data/features/idf_{0}_all/idf_{0}_train_set_1.csv'.format(fea))
        df1 = df[df['frequency'] > 0.05]
        # df2 = df[df['frequency'] < 0.10]
        DF = pd.DataFrame(columns=['id','frequency','IDF','TF_IDF'])
        if(fea == 'article'):
            df1 = df1[0:100]
            # df2 = df2[0:80]
            DF = DF.append(df1)
            # DF = DF.append(df2)
        else:
            df1 = df1[0:100]
            # df2 = df2[0:80]
            DF = DF.append(df1)
            # DF = DF.append(df2)
        for i in range(2, 20):
            df1 = pd.read_csv('../data/features/idf_{0}_all/idf_{0}_train_set_{1}.csv'.format(fea, str(i)))
            df2 = df1[df1['frequency'] > 0.05]
            df3 = df1
            Df1 = pd.DataFrame(columns=['id', 'frequency', 'IDF', 'TF_IDF'])
            if (fea == 'article'):
                df2 = df2[0:100]
                df3 = df3[0:10]
                Df1 = Df1.append(df2)
                Df1 = Df1.append(df3)
            else:
                df2 = df2[0:100]
                df3 = df3[0:10]
                Df1 = Df1.append(df2)
                Df1 = Df1.append(df3)
            DF = DF.append(Df1)
        print(DF)
        DF = DF.drop_duplicates(subset=['id'],keep='first')
        print(DF)
        DF.to_csv('../data/features/idf_{0}.csv'.format(fea), index=False, header=True)

    print('train_article is finished')
