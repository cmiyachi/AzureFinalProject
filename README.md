# Machine Learning Pipeline: Optimization using the Azure platform

## Overview
This project compares two pipeline optimizations.  The first one uses Hyperdrive (a python package that automates picking the best hyperparameters for your model) and a Scikit-learn model (regression).  The second one uses Azure and AutoML.  The models are compared for accuracy. 

## The Model Data
The data has customer observations (32,950) from a direct marketing campaign of a Portuguese banking institution. 

1. age (numeric)
1. job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
1. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
1. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
1. default: has credit in default? (categorical: 'no','yes','unknown')
1. housing: has housing loan? (categorical: 'no','yes','unknown')
1. loan: has personal loan? (categorical: 'no','yes','unknown')
1. contact: contact communication type (categorical: 'cellular','telephone')
1. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
1. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
1. duration: last contact duration, in seconds (numeric). 
1. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
1. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
1. previous: number of contacts performed before this campaign and for this client (numeric)
1. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
1. emp.var.rate: employment variation rate - quarterly indicator (numeric)
1. cons.price.idx: consumer price index - monthly indicator (numeric)
1. cons.conf.idx: consumer confidence index - monthly indicator (numeric)
1. euribor3m: euribor 3 month rate - daily indicator (numeric)
1. nr.employed: number of employees - quarterly indicator (numeric)

**Output **
1. y - has the client subscribed a term deposit? (binary: 'yes','no')


**Solution**

Because we have to predict yes or no (0 or 1), a binary classification model is used here to determine if the client will subscribe to a term deposit. The `VotingEnsemble` model by the Azure AutoML run with an Accuracy of `0.91721` which was better than logistic regression model optimized by `Hyperdrive`.


## Scikit-learn Pipeline

![Cluster Image](images/Flow.jpg)

When we ran the experiment we gave the following parameters:
ps = RandomParameterSampling({
    '--C': uniform(0.001, 1.0),
    '--max_iter': choice(0, 10, 50, 100, 150, 200)
})

When the model is run and the most accurate run is selected, here is the output:

['--C', '0.4725712603502653', '--max_iter', '200']

Thus the Regularization strength was 0.47 and the Number of iterations was 200 for the best run. 

### Logistic Regression + Hyperdrive Setup

1. A skeleton of the python `train.py` script which creates the Tabular dataset. This also contains data cleansing and the model setup for Scikit.

1. Also a skeleton of the Hyperdrive configuration settings which was fleshed out to call`train.py`.

1. The run was done on an Azure ML compute cluster. Data is imported in csv format, data is made tabular, cleansed, and split.

1. The hyperparameters are configured and the data is fed to the Scikit-learn Logistic Regression model.  

1. Different combination of hyperameter values are used to train the model. Using the test data, the Hyperdrive creates Regression models and the accuracy are logged.

1. The model with the highest accuracy is then saved.


### AutoML Configuration
1. The AutoML run is set up with a Tabular Dataset and is imported in csv format, data is made tabular, cleansed, and split.

1. The training configuration is set to `classification` and the label (dependent variable) specified.

1. The kfold `n_cross_validation` value is set, and `primary_metric` is set as `accuracy`.

1. Then run teh AutoML.

1. AutoML picks different hyperparameters and algorithms. 

1. The most accurate model from AutoML run is obtained.

**Benefits?**

Random Parameter Sampling was chosen as the Hyperparameter tuning algorithm. The benefit of this search algorithm is that it provides a good model by randomly selecting hyperparameter values from a defined search space. This makes it less computational intensive and much faster. It also supports early stoping policy.

**What are the benefits of the early stopping policy?**

If a run is performing badly it stops and no compute time is wasted. This flow use the Bandit policy which stops a run if it underperforms the best run by a defined value called "Slack". The "Bandit policy" was chosen because top performance runs are kept until the end. This run used the following arguments evaluation_interval=3, slack_factor=0.1, delay_evaluation=3 - if the best AUC reported for a run is Y, then the value (Y + Y * 0.1) to 0.9, and if smaller, cancels the run. If delay_evaluation = 3, then the first time the policy will be applied is at interval 3, so it gets aplied early on. 

## AutoML

AutoML generates different models and hyperparameters automatically for all the different algorithms.  AutoML tries 17 different algorithms combined with scaling and normalization and ensemble methods.  The run defines how long to try for and then the best model is chosen. Note that this run is set to timeout after 30 minutes. 

When the print_model is run on the best model found, we see the following parameters:
 <code>
    
datatransformer
{'enable_dnn': None,
 'enable_feature_sweeping': None,
 'feature_sweeping_config': None,
 'feature_sweeping_timeout': None,
 'featurization_config': None,
 'force_text_dnn': None,
 'is_cross_validation': None,
 'is_onnx_compatible': None,
 'logger': None,
 'observer': None,
 'task': None,
 'working_dir': None}

prefittedsoftvotingclassifier
{'estimators': ['0', '1', '13', '9', '3', '12'],
 'weights': [0.5333333333333333,
             0.2,
             0.06666666666666667,
             0.06666666666666667,
             0.06666666666666667,
             0.06666666666666667]}

0 - maxabsscaler
{'copy': True}

0 - lightgbmclassifier
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'learning_rate': 0.1,
 'max_depth': -1,
 'min_child_samples': 20,
 'min_child_weight': 0.001,
 'min_split_gain': 0.0,
 'n_estimators': 100,
 'n_jobs': 1,
 'num_leaves': 31,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.0,
 'reg_lambda': 0.0,
 'silent': True,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

1 - maxabsscaler
{'copy': True}

1 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 1,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 3,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': 1,
 'nthread': None,
 'objective': 'binary:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 1,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

13 - minmaxscaler
{'copy': True, 'feature_range': (0, 1)}

13 - sgdclassifierwrapper
{'alpha': 4.693930612244897,
 'class_weight': 'balanced',
 'eta0': 0.001,
 'fit_intercept': False,
 'l1_ratio': 0.3877551020408163,
 'learning_rate': 'constant',
 'loss': 'squared_hinge',
 'max_iter': 1000,
 'n_jobs': 1,
 'penalty': 'none',
 'power_t': 0.3333333333333333,
 'random_state': None,
 'tol': 0.001}

9 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': True,
 'with_std': True}

9 - sgdclassifierwrapper
{'alpha': 7.346965306122448,
 'class_weight': None,
 'eta0': 0.001,
 'fit_intercept': True,
 'l1_ratio': 0.8979591836734693,
 'learning_rate': 'constant',
 'loss': 'modified_huber',
 'max_iter': 1000,
 'n_jobs': 1,
 'penalty': 'none',
 'power_t': 0.6666666666666666,
 'random_state': None,
 'tol': 0.01}

3 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': True,
 'with_std': True}

3 - sgdclassifierwrapper
{'alpha': 1.4286571428571428,
 'class_weight': None,
 'eta0': 0.01,
 'fit_intercept': True,
 'l1_ratio': 0.7551020408163265,
 'learning_rate': 'constant',
 'loss': 'log',
 'max_iter': 1000,
 'n_jobs': 1,
 'penalty': 'none',
 'power_t': 0.4444444444444444,
 'random_state': None,
 'tol': 0.001}

12 - robustscaler
{'copy': True,
 'quantile_range': [10, 90],
 'with_centering': False,
 'with_scaling': False}

12 - extratreesclassifier
{'bootstrap': False,
 'ccp_alpha': 0.0,
 'class_weight': 'balanced',
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'sqrt',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 0.06157894736842105,
 'min_samples_split': 0.10368421052631578,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 50,
 'n_jobs': 1,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}
 </code>

The VotingEnsemble is tree based so notice that some of the paramters are with regards to "min_sample_leaf" and "min_samples_split" would indicate how many samples to make a split. 

## Pipeline Differences

The best model  was `VotingEnsemble` using AutoML (accuracy .9166). It was slightly better than `Logistic Regression + Hyperdrive` model (accuracy .9144).  
The AutoML took much longer to execute. Voting Ensemble is an ensemble algorithm which combines multiple models to achieve higher performance than the single models individually and uses the weighted average of predicted class probabilities. 

## Future work

As we did in class, the hyperdrive hyperparameter search algorithms(GridParameterSampling or BayesianParameterSampling) could be varied to see if the performance improves. I also think working iwth Deep Neural Networks (DNNs), and Deep Learning could improve performance. Note that on this run, there was a message that the data set was imbalanced.
We could consider other metrics such as AUC, Precision, or Recall instead of Accuracy. 


## Cluster clean up
See `aml_compute.delete()` in the code
