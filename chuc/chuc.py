import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier, XGBRegressor
from pylift.eval import UpliftEval
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, roc_auc_score
import warnings
import scipy as sc
warnings.filterwarnings("ignore")


class Uplift:
    
    def getTrainCols(self, df_temp):
        allCols=list(df_temp.columns)
        allCols.remove(self.outcome)
        allCols.remove(self.treatmentLabel)
        trainCols=allCols
        return trainCols
    
    def getBestXGBParams(self, n_jobs, n_iter, cv, scoring):
        parameters_xgb=self.param_search_space
        rsearch = RandomizedSearchCV(self.propensityAlgo(), parameters_xgb, scoring=scoring, n_jobs= n_jobs , n_iter=n_iter, cv=cv)
        X=self.uplift_train[self.trainCols]
        Y=self.uplift_train[self.outcome]        
        rsearch.fit(X,Y)
        return rsearch
    
    def group_scores(self, rank,nGroups):
        labels= [ 'group_'+ str(x) for x in range(nGroups)]
        series, bins = pd.qcut(rank.rank(method='first'), nGroups,labels = labels ,retbins=True, duplicates='drop')
        return series
    
    def getPropensityModel(self):
        if self.param_search_space is None:
            model_propensity=self.propensityAlgo()
            model_propensity.fit(self.uplift_train[self.uplift_train[self.treatmentLabel]==0][self.trainCols],
                  self.uplift_train[self.uplift_train[self.treatmentLabel]==0][self.outcome])  
        else:
            if self.propensityAlgo==XGBRegressor:
                scoring = 'neg_mean_absolute_error'
            else:
                scoring='roc_auc'
            rsearch=self.getBestXGBParams(n_jobs=50, n_iter=10, cv=10, scoring=scoring)     
            best_params=rsearch.best_params_
            print(best_params)
            model_propensity=self.propensityAlgo(params=best_params)
            model_propensity.fit(self.uplift_train[self.uplift_train[self.treatmentLabel]==0][self.trainCols],
                  self.uplift_train[self.uplift_train[self.treatmentLabel]==0][self.outcome])
        
        if self.propensityAlgo==XGBRegressor:
            self.uplift_train['propensity']=model_propensity.predict(self.uplift_train[self.trainCols])
        else:
            self.uplift_train['propensity']=model_propensity.predict_proba(self.uplift_train[self.trainCols])[:,1]
            
        self.uplift_train['rank']=self.uplift_train['propensity'].rank()
        self.uplift_train['rank_bins']=self.group_scores(self.uplift_train['rank'] ,nGroups=10)
        return model_propensity
    
    def generateUplift(self, df_temp):
        test=df_temp[df_temp[self.treatmentLabel]==1]
        control=df_temp[df_temp[self.treatmentLabel]==0]
        test_grouped=test.groupby(['rank_bins'])[self.outcome].agg(['mean']).reset_index()
        control_grouped=control.groupby(['rank_bins'])[self.outcome].agg(['mean']).reset_index()
        groupsMerged=pd.merge(left = test_grouped, right = control_grouped, how = 'inner', 
                              on='rank_bins', suffixes=('_treatment','_holdout'))
        groupsMerged['uplift']=groupsMerged['mean_treatment']-groupsMerged['mean_holdout']
        df_temp['uplift']=pd.merge(df_temp, groupsMerged, on ='rank_bins')['uplift']
        df_temp['uplift'].fillna(df_temp['uplift'].mean(), inplace= True)
        return groupsMerged
    
    def generateUpliftModel(self):
        holdoutCr=[]
        uplift=[]
        split_array=[]
        split=StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=200)
        for index1, index2 in  split.split(self.uplift_train, self.uplift_train[[self.treatmentLabel]]):
            split_array.append(index1)
            split_array.append(index2)
        for ind in range(len(split_array)):
            df_uplift=self.uplift_train.loc[split_array[ind]]
            groupsMerged=self.generateUplift(df_uplift)
            holdoutCr+=list(groupsMerged.mean_holdout)
            uplift+=list(groupsMerged.uplift)
        x=pd.DataFrame(holdoutCr)
        y=pd.DataFrame(uplift)
        x.columns=['propensity']
        y.columns=['uplift']
        self.propensity=x
        self.uplift=y
        model_uplift = XGBRegressor()
        model_uplift.fit(x,y)
        return model_uplift

    def getModels(self):
        self.model_propensity= self.getPropensityModel()
        self.model_uplift=self.generateUpliftModel()
        
    def scoreUplift(self, df_temp, is_test):
        if is_test:
            if self.propensityAlgo==XGBRegressor:
                df_temp['propensity']=self.model_propensity.predict(df_temp[self.trainCols])
            else:
                df_temp['propensity']=self.model_propensity.predict_proba(df_temp[self.trainCols])[:,1]
                
        x=pd.DataFrame(df_temp['propensity'])
        df_temp['uplift']=self.model_uplift.predict(x)
        df_temp['uplift'].fillna(df_temp['uplift'].mean(), inplace= True)
        return df_temp     
    
    def fit(self):
        self.getModels()
        self.uplift_train=self.scoreUplift(self.uplift_train, 0)
 
    def plotQini(self, df_temp):
        up_eval=UpliftEval(df_temp[self.treatmentLabel], df_temp[self.outcome], df_temp.uplift, n_bins=20)
        return up_eval

    def plotQiniTrain(self):
        up_eval=self.plotQini(self.uplift_train)
        up_eval.plot_aqini()
    
    def getDiagnostics(self):
        d={'propensity':self.propensity['propensity'], 'uplift': self.uplift['uplift']}
        df_temp=pd.DataFrame(d)
        df_temp.head()
        ax=sns.scatterplot(x='propensity', y='uplift', data=df_temp)
        df_temp['y_p']=self.model_uplift.predict(pd.DataFrame(df_temp['propensity']))
        ax.set_title('Propensity vs Uplift - Observed')
        print('**** Quality of Propensity Fit ****')
        if self.propensityAlgo==XGBRegressor:
            print('R-Squared on Train set:',r2_score(self.uplift_train[self.outcome],self.uplift_train['propensity'])  )
        else:
            print('ROC on Train set:', roc_auc_score(self.uplift_train[self.outcome],self.uplift_train['propensity']))
        
        return ax
        
    def predictUplift(self, df_test):
        df_temp=df_test.copy()
        df_temp=self.scoreUplift(df_temp, 1)
        return df_temp['uplift']
        
 
    def __init__(self, df_temp, treatmentLabel, outcome, param_search_space=None):
        
        self.treatmentLabel=treatmentLabel
        self.outcome=outcome
        self.trainCols=self.getTrainCols(df_temp)
        
        if df_temp[outcome].nunique() > 2:
            self.propensityAlgo=XGBRegressor
        else:
            self.propensityAlgo=XGBClassifier
        self.uplift_train=df_temp
        self.param_search_space=param_search_space
