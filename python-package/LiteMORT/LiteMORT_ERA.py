#Exploratory result analysis
import math
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import gc
import time
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

def auc_u_test(vec, len_A, len_B):
  rank_value = rankdata(vec)
  rank_sum = sum(rank_value[0:len_A])
  u_value = rank_sum - (len_A*(len_A+1))/2
  auc = u_value / (len_A * len_B)
  if auc < 0.50:
    auc = 1.0 - auc
  return(auc)

# from https://gist.github.com/mattsgithub/dedaa017adc1f30d9833175a5c783221
def roc_auc_alternative(y_true, y_score):
    # Total number of observations
    N = y_true.shape[0]
    I = np.arange(1, N + 1)
    N_pos = np.sum(y_true)
    N_neg = N - N_pos
    I = y_score.argsort()[::-1][:N]
    y_pred = y_true[I]
    I = np.arange(1, N + 1)
    return 1. + ((N_pos + 1.) / (2 * N_neg)) - (1. / (N_pos * N_neg)) * I.dot(y_pred)

def Robert_M_Johnson_test( ):
    np.random.seed(42)
    N = np.arange(start=20, stop=1000000, step=10000)

    t_sklearn = []
    t_dot = []
    for n in N:
        N_pos = np.random.randint(low=1, high=n + 1)
        y_true = np.concatenate((np.ones(N_pos), np.zeros(n - N_pos)))
        random.shuffle(y_true)
        y_true = np.array([0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0.])
        y_score = np.random.random(size=n)

        # Timeit
        t0 = time.time()
        y1 = roc_auc_score(y_true=y_true, y_score=y_score)
        t1 = time.time()
        t_sklearn.append(t1 - t0)

        # Timeit
        t0 = time.time()
        y2 = roc_auc_alternative(y_true=y_true, y_score=y_score)
        t1 = time.time()
        t_dot.append(t1 - t0)

        # Proves their equality
        # Raises error if not almost equal (up to 14 decimal places)
        np.testing.assert_almost_equal(y1, y2, decimal=14)

class Feature_Importance(object):
    def __init__(self, columns):
        self.columns = columns
        self.df = pd.DataFrame()

    def OnFold(self,fold,f_importance):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = self.columns
        fold_importance_df["importance"] = f_importance
        fold_importance_df["fold"] = fold
        self.df = pd.concat([self.df, fold_importance_df], axis=0)

    def SomePlot(self):
        cols = (self.df[["Feature", "importance"]].groupby("Feature").mean()
            .sort_values(by="importance", ascending=False)[:32].index)
        best_features = self.df.loc[self.df.Feature.isin(cols)]
        plt.figure(figsize=(14, 25))
        sns.barplot(x="importance",                y="Feature",
                    data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances.png')
        plt.show(block=True)

def ERA_pair(features,train,test_0,predict_0, target_0):
    predict = pd.Series(predict_0)
    target = target_0.reset_index(drop=True)
    err = (predict-target).abs()
    a0,a1=err.min(),err.max()
    thrsh = a0+(a1-a0)/10
    idx_1 = err.index[err > thrsh]
    idx_2 = err.index[err <= thrsh]
    test = test_0.reset_index(drop=True)
    df1 = test[test.index.isin(idx_1)][features]
    df2 = test[test.index.isin(idx_2)][features]
    df1.fillna(0.0, inplace=True)
    df2.fillna(0.0, inplace=True)
    g = sns.pairplot(df1)
    g.fig.suptitle("df={} err=[{:.3g}-{:.3g}]".format(df1.shape,thrsh,a1))
    g = sns.pairplot(df2)
    g.fig.suptitle("df={} err=[{:.3g}-{:.3g}]".format(df2.shape, a0,thrsh))
    plt.show()
    plt.show(block=True)
    #plt.close()
    del df1,df2
    gc.collect()
    return

#   'h_mean','log_mean'必须不是0
def df_mix_(df,cols,alg='exp_mean'):
    mix_lg, mix_hm = 0, 0
    if alg=='log_mean':
        gc.collect()
    elif alg == 'exp_mean':
        for col in cols:
            mix_hm += np.exp(df[col])
        mix_hm = np.log(mix_hm)
        df['mix'] = mix_hm
        gc.collect()
    elif alg=='h_mean':
        for col in cols:
            mix_hm += 1 / df[col]
        mix_hm = 1/mix_hm
        df['mix'] = mix_hm
        gc.collect()
    else:
        df['mix'] = df[cols].max(axis=1)
    return df['mix']

def cys_mix_ID_TRAGET_(ID,TARGET,path,files,alg='h_mean'):
    #path='H:/Project/fraud_click/bagging/'
    #files = [path+'{{{[H]_7_0.05.txt}}}_cys_.csv',path+'{{{[H]_8_eta.txt}}}_cys_.csv',path+'{{{[H]_9_eta.txt}}}_cys_.csv']
    mix_lg,mix_hm=0,0
    #alg='log_mean'              #效果很好，令人吃惊
    #alg='h_mean'                #harmonic mean
    # alg='max_out'                #
    # alg='log_rank_mean'        #可以试试
    out = '{}[{}]_BAG{}.csv'.format(path,alg,len(files))
    df = pd.DataFrame()
    cols = []
    if alg=='log_mean':
        for idx, fp in enumerate(files):
            print('====== Load {}...'.format(fp))
            tmp = pd.read_csv(fp, nrows=10000)  # , nrows=10000
            mix_lg += np.log(tmp.TARGET)
        df[ID] = tmp[ID]
        mix_lg = np.exp(mix_lg / len(files))
        df[TARGET] = mix_lg
        del tmp
        gc.collect()
    elif alg=='h_mean':
        for idx, fp in enumerate(files):
            print('====== Load {}...'.format(fp))
            tmp = pd.read_csv(fp)  #, nrows=10000
            mix_hm += 1 / (tmp.TARGET)
        df[ID] = tmp[ID]
        mix_hm = 1/mix_hm
        df[TARGET] = mix_hm
        del tmp
        gc.collect()
    else:
        df=pd.DataFrame()
        cols=[]
        for idx, fp in enumerate(files):
            print('====== Load {}...'.format(fp))
            tmp = pd.read_csv(fp)       # , nrows=10000
            title = 'att_{}'.format(idx)
            cols.append(title)
            df[title] = tmp[TARGET]
        df[ID] = tmp[ID]
        df[TARGET] = df[cols].max(axis=1)
        out = path+'{{{'+'maxout'+'}}}_bag.csv'
    nN = df.isnull().sum().sum()
    print( '======{} out={} shape={},NAN={} ...\n{}'.format(alg,out,df.shape,nN,df.head()) )
    df[[ID, TARGET]].to_csv(out, index=False,float_format='%.8f')
    print( '======{} ... OK\n'.format(alg,out,df.shape,nN) )

if __name__=='__main__':
    Robert_M_Johnson_test()