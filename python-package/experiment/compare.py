'''

'''
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import time
sys.path.insert(0, './python-package/')
import lightgbm
if True:    #liteMORT
    sys.path.insert(1, 'E:/LiteMORT/python-package/')
    import litemort
    from litemort import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch, torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.model_selection import KFold
from qhoptim.pyt import QHAdam
from tabular_data import *
import multiprocessing
from multiprocessing import Process
import psutil
#You should set the path of each dataset!!!
data_root = "F:/Datasets/"


def Monitor_Process(pid,isExitMemCheck,process_info):     #https://blog.csdn.net/u012234115/article/details/102687721
    mem_info = psutil.virtual_memory()
    p = psutil.Process(pid)
    process_info["mem_peak"] = 0.0
    process_info["cpu_peak"] = 0.0
    interval = 1 # polling seconds
    #with open("process_monitor_" + p.name() + '_' + str(pid) + ".csv", "a+") as f:
        #f.write("time,cpu%,mem%\n") # titles
    while isExitMemCheck.value==0:
        current_time = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))
        cpu_percent = p.cpu_percent()
        mem_percent = p.memory_percent()
        mem_peak = mem_info.total/100.0*mem_percent/1.0e6
        line = current_time + ',' + str(cpu_percent) + ',' + str(mem_percent)
        print (line)
        process_info["mem_peak"] = max(process_info["mem_peak"],mem_peak)
        process_info["cpu_peak"] = max(process_info["cpu_peak"],cpu_percent)
        #f.write(line + "\n")
        time.sleep(interval)
    process_info["memory"] = process_info["mem_peak"]
    print(f"exit!!!")


def GetImportLibs(libs_0):
    libs = []
    for lib in libs_0:
        if lib in sys.modules:
            modul = globals()[lib]
            name = modul.__name__
            print(f"****** {lib}={modul.__version__}")            
            libs.append(lib)
        else:
            pass
        
    return libs

class Experiment:
    def __init__(self,libs, config):
        self.profile = LiteMORT_profile()
        self.model = "QForest"
        #self.tree_type = tree_type
    
        self.GBDT_libs = ["lightgbm","litemort","xgboost","catboost"]
        self.config = config
        self.check_path(["result_path"])
        
        self.data_trans = config['data_trans'] if 'data_trans' in config else [""]
        self.libs = libs
    
    def check_path(self,path_items):
        for item in path_items:
            if item not in self.config:
                continue
            path = self.config[item]
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            assert os.path.exists(path)


    def InitExperiment(self,config,fold_n):
        pass
        config.experiment = f'{config.data_set}_{config.model_info()}_{fold_n}'   #'year_node_shallow'
        #experiment = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}_{:0>2d}'.format(experiment, *time.gmtime()[:5])
        #visual = quantum_forest.Visdom_Visualizer(env_title=config.experiment)
        visual = quantum_forest.Visualize(env_title=config.experiment)
        visual.img_dir = "{result_path}/images/"
        print("experiment:", config.experiment)
        log_path=f"logs/{config.experiment}"
        if os.path.exists(log_path):        #so strange!!!
            import shutil
            print(f'experiment {config.experiment} already exists, DELETE it!!!')
            shutil.rmtree(log_path)
        return config,visual
    
    def GBDT_learn(self,lib,num_rounds = 100000,bf=1,ff=1,fold_n=0):  
        info={}
        assert hasattr(self,'data')
        data = self.data
        result_path = self.config["result_path"]      
        nFeatures = data.X_train.shape[1]
        early_stop = 100;    verbose_eval = 20
        
        #lr = 0.01;   
        bf = bf;    ff = ff

        if data.problem()=="classification":
            metric = 'auc'       #"rmse"
            params = {"objective": "binary", "metric": metric,'n_estimators': num_rounds,
            "bagging_fraction": bf, "feature_fraction": ff,'verbose_eval': verbose_eval, "early_stopping_rounds": early_stop, 'n_jobs': -1, 
                }
        else:
            metric = 'l2'       #"rmse"
            params = {"objective": "regression", "metric": metric,'n_estimators': num_rounds,
                "bagging_fraction": bf, "feature_fraction": ff, 'verbose_eval': verbose_eval, "early_stopping_rounds": early_stop, 'n_jobs': -1,
                }
        print(f"====== GBDT_test\tparams={params}")
        X_train, y_train = data.X_train, data.y_train
        X_valid, y_valid = data.X_valid, data.y_valid
        X_test, y_test = data.X_test, data.y_test
        if not np.isfortran(X_train):   #Very important!!! mort need COLUMN-MAJOR format
            X_train = np.asfortranarray(X_train)
            X_valid = np.asfortranarray(X_valid)
        #X_train, X_valid = pd.DataFrame(X_train), pd.DataFrame(X_valid)
        print(f"GBDT_test\ttrain={X_train.shape} valid={X_valid.shape}")
        #print(f"X_train=\n{X_train.head()}\n{X_train.tail()}")
        if lib == 'litemort':
            params['verbose'] = 667
            model = LiteMORT(params).fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            #y_pred_valid = model.predict(X_valid)
            #y_pred = model.predict(X_test)

        elif lib == 'lightgbm':
            if data.problem()=="classification":
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=min(num_rounds//10,1000))
            pred_val = model.predict(data.X_test)
            #plot_importance(model)
            lgb.plot_importance(model, max_num_features=32)
            plt.title("Featurertances")
            plt.savefig(f"{result_path}/{self.data_name}_feat_importance_.jpg")
            #plt.show(block=False)
            plt.close()

            fold_importance = pd.DataFrame()
            fold_importance["importance"] = model.feature_importances_
            fold_importance["feature"] = [i for i in range(nFeatures)]
            fold_importance["fold"] = fold_n
            #fold_importance.to_pickle(f"{result_path}/{dataset}_feat_{fold_n}.pickle")
            print('best_score', model.best_score_)
            acc_train,acc_=model.best_score_['training'][metric], model.best_score_['valid_1'][metric]
            info["feature_importance"] = fold_importance
        else:
            raise f"{lib} is XXX"
        if data.X_test is not None:
            pred_val = model.predict(data.X_test)
            info["accuracy"] = ((data.y_test - pred_val) ** 2).mean()
            print(f'====== GBDT_learn={lib}\tstep: test={data.X_test.shape} ACCU@Test={info["accuracy"]:.5f}')
        return info
    
    def run_KFold(self):
        folds = KFold(n_splits=nFold, shuffle=True)
        index_sets=[]
        for fold_n, (train_index, valid_index) in enumerate(folds.split(data.X)):
            index_sets.append(valid_index)
        for fold_n in range(len(index_sets)):
            config, visual = InitExperiment(config, fold_n)
            train_list=[]
            for i in range(nFold):
                if i==fold_n:           #test
                    continue
                elif i==fold_n+1:       #valid                
                    valid_index=index_sets[i]
                else:
                    train_list.append(index_sets[i])
            train_index=np.concatenate(train_list)
            print(f"train={len(train_index)} valid={len(valid_index)} test={len(index_sets[fold_n])}")

            data.onFold(fold_n,config,train_index=train_index, valid_index=valid_index,test_index=index_sets[fold_n],pkl_path=f"{data_root}{dataset}/FOLD_{fold_n}.pickle")
            Fold_learning(fold_n,data,config,visual)
        pass

    def learn(self,lib):
        self.profile.Snapshot("fit_0")
        result = {"method":lib}
        t0 = time.time()
        if lib in self.GBDT_libs:  
            result.update(self.GBDT_learn(lib))   #
            #result.update({"accuracy":0.0})
        elif lib == "QForest":
            pass
            if config.feat_info == "importance":
                feat_info = get_feature_info(data,fold_n)            
            else:
                feat_info = None
            result = NODE_test(data,fold_n,config,visual,feat_info)
        elif lib == "LinearRegressor":
            pass
            model = quantum_forest.Linear_Regressor({'cascade':"ridge"})
            result = model.fit((data.X_train, data.y_train),[(data.X_test, data.y_test)])
        result["time"] = time.time()-t0
        
        self.profile.Stat("fit_0","fit_1",dump=False)     #https://psutil.readthedocs.io/en/latest/
        result['virtual memory']=self.profile.memory_info['virtual memory']         #total amount of virtual memory used by the process
        result['physical memory']=self.profile.memory_info['physical memory']       #non-swapped physical memory a process has used  
        #result["memory"] = result['physical memory']     
        return result
    
    def Fold_learning(self,fold_n,data,config,visual):
        t0 = time.time()
        if config.model=="QForest":
            if config.feat_info == "importance":
                feat_info = get_feature_info(data,fold_n)            
            else:
                feat_info = None
            accu,_ = NODE_test(data,fold_n,config,visual,feat_info)
        elif config.model=="GBDT":
            accu,_ = GBDT_test(data,fold_n)
        else:        #"LinearRegressor"    
            model = quantum_forest.Linear_Regressor({'cascade':"ridge"})
            accu,_ = model.fit((data.X_train, data.y_train),[(data.X_test, data.y_test)])

        print(f"\n======\n====== Fold_{fold_n}\tACCURACY={accu:.5f},time={time.time() - t0:.2f} ====== \n======\n")
        return

class Compare(Experiment):
    def __init__(self,libs,config ):
        super().__init__(libs,config)

        pass
    
    #https://stackoverflow.com/questions/38807895/seaborn-multiple-barplots/38808042
    def plot_factor(self,title_,df,feat_X,feat_Y,feat_Group,ext_info=""):
        label='%.4f'
        if feat_Y=="memory":
            label='%.2f M'
        elif feat_Y=="accuracy":
            label='AUC=%.4f'
        elif feat_Y=="time":
            label='%.2f s'
        df_1 = df[[feat_X,feat_Y,feat_Group]]
        #df_2 = pd.melt(df_1, id_vars="method", var_name="dataset", value_name="accuracy")
        sns.factorplot(feat_X, feat_Y, col=feat_Group, data=df_1, kind="bar")
        # Get current axis on current figure
        y_max = df_1[feat_Y].max() 
                
        axes = plt.gcf().get_axes()
        for ax in axes:
            ax.set_ylim([0, y_max*1.2])    
            # Iterate through the list of axes' patches
            for p in ax.patches:            
                height = p.get_height() # height of bar, which is basically the data value
                ax.text(p.get_x() + p.get_width()/2., height, label%height, 
                        fontsize=16, color='red', ha='center', va='bottom')

        plt.savefig(f"{title_}___{feat_Y}@{feat_X}@{feat_Group}.jpg")
        plt.show()
        plt.close()        
    
    def plot_compare(self,title,results):
        df = pd.DataFrame(results["list"])
        
        self.plot_factor(title,df,"method","memory","dataset")
        self.plot_factor(title,df,"method","accuracy","dataset")      
        self.plot_factor(title,df,"method","time","dataset")     

    def run(self,data_list,exp_name="cys_",just_plot=False):
        pkl_path=f"./data/{exp_name}_{data_list}_compare_.pickle"
        plot_title = f"./data/{data_list}"
        if just_plot and os.path.isfile(pkl_path): 
            with open(pkl_path, "rb") as fp:
                results = pickle.load(fp)
            self.plot_compare(plot_title,results)
            return    

        pid = os.getpid()
        date = '{}.{:0>2d}.{:0>2d}_{:0>2d}_{:0>2d}'.format(*time.gmtime()[:5])
        results={"date":date,"list":[]}
        for data_name in data_list:
            self.data_name = data_name
            for data_trans in self.data_trans:
                self.data = TabularDataset(data_name,data_path=data_root, random_state=1337)
                self.data.onTrans(data_trans,self.config,pkl_path=f"{data_root}{data_name}/{data_trans}_.pickle")
                key=(data_trans,)
                for lib in libs:
                    key = key+(lib,)    
                    isExitMemCheck = multiprocessing.Value("i",0)
                    process_info=multiprocessing.Manager().dict()
                    p = Process(target=Monitor_Process,args=(pid,isExitMemCheck,process_info))
                    p.start()                    
                    result = self.learn(lib) 
                    isExitMemCheck.value = 1  
                    p.join()
                    #mem_info = mem_info["peak"]                    print(mem_info)
                    result.update(process_info)
                    result.update({"dataset":data_name})
                    results[key] = result                     
                    results["list"].append(result)
                
        if pkl_path is not None:
                with open(pkl_path, "wb") as fp:
                    pickle.dump(results,fp)
        self.plot_compare(plot_title,results)
    

def get_feature_info(data,fold_n):
    pkl_path = f"{result_path}/{dataset}_feat_info_.pickle"
    nSamp,nFeat = data.X_train.shape[0],data.X_train.shape[1]
    if os.path.isfile(pkl_path):
        feat_info = pd.read_pickle(pkl_path)
    else:
        #fast GBDT to get feature importance
        nMostSamp,nMostFeat=100000.0,100.0
        bf = 1.0 if nSamp<=nMostSamp else nMostSamp/nSamp
        ff = 1.0 if nFeat<=nMostFeat else nMostFeat/nFeat
        accu,feat_info = GBDT_test(data,fold_n,num_rounds=2000,bf = bf,ff = ff)
        with open(pkl_path, "wb") as fp:
            pickle.dump(feat_info, fp)

    importance = torch.from_numpy(feat_info['importance'].values).float()
    fmax, fmin = torch.max(importance), torch.min(importance)
    weight = importance / fmax
    feat_info = data.OnFeatInfo(feat_info,weight)
    return feat_info

def cascade_LR():   #意义不大
    if config.cascade_LR:
        LinearRgressor = quantum_forest.Linear_Regressor({'cascade':"ridge"})
        y_New = LinearRgressor.BeforeFit((data.X_train, data.y_train),[(data.X_valid, data.y_valid),(data.X_test, data.y_test)])
        YY_train = y_New[0]
        YY_valid,YY_test = y_New[1],y_New[2]
    else:
        YY_train,YY_valid,YY_test = data.y_train, data.y_valid, data.y_test
    return YY_train,YY_valid,YY_test

def VisualAfterEpoch(epoch,visual,config,mse):
    if visual is None:
        if config.plot_train:
            clear_output(True)
            plt.figure(figsize=[18, 6])
            plt.subplot(1, 2, 1)
            plt.plot(loss_history)
            plt.title('Loss')
            plt.grid()
            plt.subplot(1, 2, 2)
            plt.plot(mse_history)
            plt.title('MSE')
            plt.grid()
            plt.show()
    else:
        visual.UpdateLoss(title=f"Accuracy on \"{dataset}\"",legend=f"{config.experiment}", loss=mse,yLabel="Accuracy")




if __name__ == "__main__":
    libs = GetImportLibs(["lightgbm","litemort","xgboost","catboost","node","quantumforest"])    
    #data = TabularDataset("CLICK",data_path=data_root, random_state=1337, quantile_transform=True, quantile_noise=1e-3)    
    config = {
        "random_state":             42,
        "device":                   ["CPU","GPU"],
        #"data_trans":               ["Quantile"],
        "cross_fold":               5,
        "result_path":              "./result/",
    }
    for lib in libs:
        config[lib] = {}

    experiment = Compare(libs,config)
    #"MICROSOFT","YAHOO","YEAR","CLICK","HIGGS"
    experiment.run(["CLICK","MICROSOFT",],just_plot=True)   
    
            



