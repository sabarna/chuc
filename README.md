# CHUC

CHUC (or Conversion Homogeneity based Uplift Computation) is a Python package to compute the incremental effect treatment (eg: Marketing Ad impression) has on a subject's probability to convert (or expected revenue).

# Training Algorithm
- Group customers based on their probability to convert using the holdout set -using a Classification (if outcome is binomial) or Regression (if outcome is continuous) model **M1** 
- Compute the empirical uplift on each of the groups
- Build a regression model, **M2**, that maps conversion to uplift based on the above observations
# Scoring Algorithm
  - Using **M1** assign every observation into a group
  - Using **M2** predict Uplift for that group

# Installation
```sh
pip install chuc
```
# Training
```sh
#df -> dataframe containing testlabel, outcome and predictors  
#treatmentLabel -> Column name of the treatmen label
#outcome -> Column name of the outcome variable

import chuc
# Instantiating the uplift object with no hyperparamter tuning
u=chuc.Uplift(df,'testgroup', 'conversion') 

#Instanting the model object to train M1 with hyperparameter tuning
p_={'max_depth':list(range(2,15,1)), 
    'min_child_weight': list(range(10,300)),
    'n_estimators':list(range(50,400)),
    'gamma': sc.stats.uniform(0.1, 0.9),
    'subsample':sc.stats.uniform(0.6, 0.4)}
u=chuc.Uplift(df,treatmentLabel= 'testgroup', outcome= 'converted',p_)

#Fitting the model
u.fit()

#Getting diagnostics plot 
u.getDiagnostics()
```

# Predicting
```sh
uplift=u.predictUplift(df_test)
```

