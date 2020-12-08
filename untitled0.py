# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:05:12 2019

@author: willh
"""
### import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
import datetime as dt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.preprocessing import LabelEncoder
import datetime as dt
from scipy.linalg.decomp_svd import LinAlgError
from imblearn.over_sampling import SMOTENC
from sklearn.pipeline import Pipeline
from sklearn_pandas import CategoricalImputer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_validate

### import data
data = pd.read_csv(r'C:\Users\willh\fs\exercise_01_train.csv')

test_data = pd.read_csv(r'C:\Users\willh\fs\exercise_01_test.csv')

# view data description
data.head()
data.describe()
data.shape
data.dtypes

def cont_and_cat(data):
    """
    Clean categorical features: 
        - x41 and x45: remove dollar signs and change data type to float
        - x34, x68, x93: fix typos.
    Create continuous and categorical columns variables.
    """
    cat_cols = data.select_dtypes(include='object').columns
    cont_cols = data.drop(columns=cat_cols).columns
    data[cat_cols].head()
    # remove dollar and percent signs and change data type to float
    data['x41'] = data['x41'].str.replace('$','').astype(float)
    data['x45'] = data['x45'].str.replace('%','').astype(float)
    # clean some of the categorical data
    data['x34'].unique()
    data['x35'].unique()
    data['x35'] = data['x35'].replace(dict.fromkeys(['thurday','thur'], 'thursday'))
    data['x35'] = data['x35'].replace({'wed':'wednesday', 'fri':'friday'})
    data['x68'].unique()
    data['x68'] = data['x68'].replace({'January':'Jan', 'sept':'Sep','Dev': 'Dec'})
    data['x93'].unique()
    data['x93'] = data['x93'].replace({'euorpe':'europe'})
    cat_cols = data.select_dtypes(include='object').columns
    cont_cols = data.drop(columns=cat_cols).columns
#    cont_cols = cont_cols[:-1]  # commenting out for test set; test set does not have 'y' column
    
    return cont_cols, cat_cols

cont_cols, cat_cols = cont_and_cat(data)

# check the nulls
data.isnull().sum()
# check for balance
data.y.value_counts()
data['y'][data['y'] == 1].count()/len(data)
# data is imbalanced- about 80/20 of dependent variable

### take a look at the data when dropping missing values
notnull = data.dropna()
notnull.shape
notnull.isnull().sum()
39198/40000
# dropped only about 3 percent of the total data

# label encode categorical data
for i in cat_cols:
    encoder = LabelEncoder()
    notnull.loc[:,i] = encoder.fit_transform(notnull.loc[:,i])
    
# take a look at correlations - if there are any noticeable correlations
corr = notnull.corr()
ax = sns.heatmap(corr,cmap="YlGnBu")

pd.plotting.scatter_matrix(notnull)
plt.show()

# take a look at the distributions 
fig = plt.figure(figsize = (20, 25))
j = 0
for i in notnull.columns[0:24]:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(notnull[i][notnull['y']==0], color='r', label = 'No')
    sns.distplot(notnull[i][notnull['y']==1], color='g', label = 'Yes')
    plt.legend(loc='best')
fig.suptitle('Data')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show(fig)

fig2 = plt.figure(figsize = (20, 25))
j = 0
for i in notnull.columns[25:49]:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(notnull[i][notnull['y']==0], color='r', label = 'No')
    sns.distplot(notnull[i][notnull['y']==1], color='g', label = 'Yes')
    plt.legend(loc='best')
fig2.suptitle('Data')
fig2.tight_layout()
fig2.subplots_adjust(top=0.95)
plt.show()

fig3 = plt.figure(figsize = (20, 25))
j = 0
for i in notnull.columns[50:74]:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(notnull[i][notnull['y']==0], color='r', label = 'No')
    sns.distplot(notnull[i][notnull['y']==1], color='g', label = 'Yes')
    plt.legend(loc='best')
fig3.suptitle('Data')
fig3.tight_layout()
fig3.subplots_adjust(top=0.95)
plt.show()

fig4 = plt.figure(figsize = (20, 25))
j = 0
for i in notnull.columns[75:99]:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(notnull[i][notnull['y']==0], color='r', label = 'No')
    sns.distplot(notnull[i][notnull['y']==1], color='g', label = 'Yes')
    plt.legend(loc='best')
fig4.suptitle('Data')
fig4.tight_layout()
fig4.subplots_adjust(top=0.95)
plt.show()

fig5 = plt.figure(figsize = (8, 6))
sns.distplot(notnull['x99'][notnull['y']==0], color='r', label = 'No')
sns.distplot(notnull['x99'][notnull['y']==1], color='g', label = 'Yes')
plt.legend(loc='best')
fig5.suptitle('Data')
fig5.tight_layout()
fig5.subplots_adjust(top=0.95)
plt.show()
# They're all normal distributions - no need for log transformations

# Begin looking at variable relationships...did not have time to continue...
fig01 = plt.figure(figsize = (20, 25))
j = 0
for i in notnull.columns[0:24]:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.scatterplot(x = 'x0', y = i, hue='y', data=notnull)
    plt.legend(loc='best')
fig01.suptitle('Data')
fig01.tight_layout()
fig01.subplots_adjust(top=0.95)
plt.show()

# separate continuous and categorical data
cat_data = data[cat_cols]
cont_data = data[cont_cols]

def outliers_and_missing(cat_data, cont_data):
    """
    - Find and replace upper and lower outliers with upper and lower limits.
    - Replace missing continuous values with mean of column.
    - Replace missing categorical values with most frequent of column.
    """
    Q1 = cont_data.quantile(0.25)
    Q3 = cont_data.quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    upper_mask = cont_data > upper_limit
    lower_mask = cont_data < lower_limit
    cont_data = cont_data.mask(upper_mask, upper_limit, axis = 1)
    cont_data = cont_data.mask(lower_mask, lower_limit, axis = 1)
    
    # impute missing data using mean for continuous variables
    cont_data.fillna(cont_data.mean(), inplace = True)
    
    # impute missing categorical data using mode
    imp34 = CategoricalImputer()
    cat_data['x34'] = imp34.fit_transform(cat_data['x34'])
    imp35 = CategoricalImputer()
    cat_data['x35'] = imp35.fit_transform(cat_data['x35'])
    imp68 = CategoricalImputer()
    cat_data['x68'] = imp68.fit_transform(cat_data['x68'])
    imp93 = CategoricalImputer()
    cat_data['x93'] = imp93.fit_transform(cat_data['x93'])

    # combine the imputed continuous and categorical data
    imp_data = pd.concat([cont_data, cat_data], axis=1)
    
    return imp_data

imp_data = outliers_and_missing(cat_data, cont_data)

def encode_cat_and_ref_df(imp_data, cat_cols):
    """
    - Label encode categorical data in preparation for over sampling (SMOTE)
    - Create categorical reference dict to map label encoded values back to original values.
    """
    # label encode categorical data
    le34 = LabelEncoder()
    imp_data['x34'] = le34.fit_transform(imp_data['x34'])
    le35 = LabelEncoder()
    imp_data['x35'] = le35.fit_transform(imp_data['x35'])
    le68 = LabelEncoder()
    imp_data['x68'] = le68.fit_transform(imp_data['x68'])
    le93 = LabelEncoder()
    imp_data['x93'] = le93.fit_transform(imp_data['x93'])
    # make a reference df for the encoded categories
    cat_ref_df = imp_data[cat_cols].copy()
    cat_ref_df['x34_orig'] = le34.inverse_transform(cat_ref_df['x34'])
    cat_ref_df['x35_orig'] = le35.inverse_transform(cat_ref_df['x35'])
    cat_ref_df['x68_orig'] = le68.inverse_transform(cat_ref_df['x68'])
    cat_ref_df['x93_orig'] = le93.inverse_transform(cat_ref_df['x93'])
    cat_ref_df = cat_ref_df.reindex(sorted(cat_ref_df.columns), axis=1)
    cat_dic = {}
    cat_dic['x34'] = dict(zip(cat_ref_df['x34'], cat_ref_df['x34_orig']))
    cat_dic['x35'] = dict(zip(cat_ref_df['x35'], cat_ref_df['x35_orig']))
    cat_dic['x68'] = dict(zip(cat_ref_df['x68'], cat_ref_df['x68_orig']))
    cat_dic['x93'] = dict(zip(cat_ref_df['x93'], cat_ref_df['x93_orig']))
    
    return imp_data, cat_dic

imp_data, cat_dic = encode_cat_and_ref_df(imp_data, cat_cols)

### no oversampling
from sklearn.pipeline import Pipeline
rf_nos = Pipeline(steps=[('random_forest', RandomForestClassifier(criterion='entropy'))])
scores = cross_val_score(rf_nos, imp_data, data.y, cv=10,scoring='accuracy')
print('Accuracy for RandomForest : ', scores.mean())

### Over sampling with smote
def smote_oversampling(imp_data, data, rand_state, categoricals=None):
    """
    - Over sample the minority category of the dependent variable (1).
    - Split the data into train and test sets.
    """
    os = SMOTENC(categorical_features = categoricals ,random_state = rand_state)
    X_train, X_test, y_train, y_test = train_test_split(imp_data, data.y, test_size = 0.2)
    columns = X_train.columns
    
    os_data_X, os_data_y = os.fit_sample(X_train, y_train)
    os_data_X = pd.DataFrame(data = os_data_X, columns = columns)
    os_data_y = pd.DataFrame(data = os_data_y, columns = ['y'])
    print('length of oversampled data is ',len(os_data_X))
    print('Number of no subscription in oversampled data ', len(os_data_y[os_data_y['y']==0]))
    print('Number of subscription', len(os_data_y[os_data_y['y'] == 1]))
    print('Proportion of no subscription data in oversampled data is ', len(os_data_y[os_data_y['y']==0])/len(os_data_X))
    print('Proportion of subscription data in oversampled data is ', len(os_data_y[os_data_y['y']==1])/len(os_data_X))
    
    return os_data_X, X_test, os_data_y, y_test

cat_feats = [96, 97, 98, 99]
rand_state = 0
os_data_X, X_test, os_data_y, y_test = smote_oversampling(imp_data, data, rand_state, categoricals=cat_feats)

### DUMMIES
def dummies(os_data_X, X_test, cat_dic, cat_cols):
    """
    - Re-encode categorical values to original string values.
    - Make dummy variables for categorical columns in train and test set.
    """
    # re-encode values of categorical features
    dummy_X_train = os_data_X.copy()
    dummy_X_test = X_test.copy()
    for key in cat_dic.keys():
        dummy_X_train[key] = dummy_X_train[key].map(cat_dic[key])
    for key in cat_dic.keys():
        dummy_X_test[key] = dummy_X_test[key].map(cat_dic[key])
    
    # make dummies for train set    
    for col in cat_cols:
        cat_list='var'+'_'+col
        cat_list = pd.get_dummies(dummy_X_train[col], prefix=col, drop_first = True)
        data1=dummy_X_train.join(cat_list)
        dummy_X_train=data1
    data_vars=dummy_X_train.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_cols]
    # final data columns will be:
    dummy_X_train=dummy_X_train[to_keep]
    
    # make dummies for test set    
    for col in cat_cols:
        cat_list='var'+'_'+col
        cat_list = pd.get_dummies(dummy_X_test[col], prefix=col, drop_first = True)
        data1=dummy_X_test.join(cat_list)
        dummy_X_test=data1
    data_vars=dummy_X_test.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_cols]
    # final data columns will be:
    dummy_X_test=dummy_X_test[to_keep]
    
    return dummy_X_train, dummy_X_test

dummy_X_train, dummy_X_test = dummies(os_data_X, X_test, cat_dic, cat_cols)

# check if length of train columns match length of test columns.  Sometimes if the test set is missing a categorical value, it will create fewer dummy variables.
def col_len_check(dummy_X_train, dummy_X_test):
    train_cols = dummy_X_train.shape[1]
    test_cols = dummy_X_test.shape[1]
    if train_cols != test_cols:
        print("X train columns don't match X test columns.  Test set is missing some categorical values. Re-run smote_oversampling()")
    else:
        print("Column lengths match.  Continue.")
col_len_check(dummy_X_train, dummy_X_test)


def plot_roc_curve(fpr, tpr):
    """
    Function for plotting AUC-ROC curve
    """
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0,1], [0,1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

###########################
### Logistic Regression ###
###########################
kfold = StratifiedKFold(n_splits=10, random_state=1)
# log reg - Cross Validation - all features
log_reg = Pipeline(steps=[('scaler',StandardScaler()),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200))])
lr_scores = cross_val_score(log_reg, dummy_X_train, os_data_y, cv=kfold,scoring='roc_auc')
print('AUC scores for Logistic Regression: ', lr_scores.mean())
# AUC Logistic Regression: 0.9315

### LR Feature Selection #1 
# Use RFE to find top 25 features
# RFE (Recursive Feature Elimination) logistic regression
lr = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200)
rfe = RFE(lr, 35)
rfe = rfe.fit(dummy_X_train, os_data_y)
print(rfe.support_)
print(rfe.ranking_)
indices = [i for i, x in enumerate(rfe.ranking_) if x == 1]
cols = dummy_X_train.iloc[:,indices].columns.tolist()

# get the column names and run the oversampling function and dummy function
cols = ['x34' if x in ('x34_Toyota','x34_bmw','x34_chrystler','x34_ford','x34_nissan','x34_tesla','x34_volkswagon')  else x for x in cols]
cols = ['x35' if x in ('x35_tuesday','x35_wednesday') else x for x in cols]
cols = ['x68' if x in ('x68_Aug','x68_July','x68_Mar','x68_Mar','x68_sept.')  else x for x in cols]
cols = ['x93' if x in ('x93_asia','x93_europe')  else x for x in cols]
cols = list(set(cols))
cat_feats=[]
categoricals = ['x34','x35','x68','x93']
for i in categoricals:
    cat_feats.append(cols.index(i))
    
lr20_X_train, lr20_X_test, lr20_y_train, lr20_y_test = smote_oversampling(imp_data[cols], data, rand_state, categoricals=cat_feats)
cat_dict20 = {}
for i in categoricals:
    cat_dict20[i] = cat_dic[i]
    cat_dict20[i] = cat_dic[i]
lr20_X_train, lr20_X_test = dummies(lr20_X_train, lr20_X_test, cat_dict20, categoricals)
col_len_check(lr20_X_train, lr20_X_test)

# run lr with new features
scaler_train = StandardScaler()
lr20_X_train = scaler_train.fit_transform(lr20_X_train)
scaler_test = StandardScaler()
lr20_X_test = scaler_test.fit_transform(lr20_X_test)
lr20 = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200)
lr20.fit(lr20_X_train, lr20_y_train)
lr20_y_pred = lr20.predict(lr20_X_test)
cmlr20 = confusion_matrix(lr20_y_test, lr20_y_pred)
lr20_score = lr20.score(lr20_X_test, lr20_y_test) # score: 0.657
lr20_prec = precision_score(lr20_y_test, lr20_y_pred) # precision: 0.3440
lr20_rec = recall_score(lr20_y_test, lr20_y_pred) # recall: 0.8611
lr20probs = lr20.predict_proba(lr20_X_test)
lr20probs = lr20probs[:, 1]
lr20auc = roc_auc_score(lr20_y_test, lr20probs) # AUC: 0.8277
print(f'AUC: {lr20auc}')
fpr, tpr, thresholds = roc_curve(lr20_y_test, lr20probs)
plot_roc_curve(fpr, tpr)

# LR Feature selection #2
# Get the Coefficients of the variables at each kfold to see the top 25 features
lr_cv_est = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200)
output_lr = cross_validate(lr_cv_est, dummy_X_train, os_data_y, cv=kfold, scoring = 'roc_auc', return_estimator =True)
feature_importances_lr = pd.DataFrame()
for idx,estimator in enumerate(output_lr['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    feature_importances_lr_temp = pd.DataFrame(abs(estimator.coef_[0]).tolist(),
                                       index = dummy_X_train.columns,
                                        columns=['Coef']).sort_values('Coef', ascending=False)
    print(feature_importances_lr_temp[:15])
    feature_importances_lr = pd.concat([feature_importances_lr,feature_importances_lr_temp])
# Group all the feature importances from all the kfolds and get top 25
lr_top25_cols = feature_importances_lr.groupby(feature_importances_lr.index).sum().sort_values('Coef', ascending=False) 

# get the column names and run the oversampling function and dummy function
lr_top25_cols.reset_index(inplace=True)
lr_top25_cols = lr_top25_cols.replace(['x34_Toyota','x34_bmw','x34_chrystler','x34_ford','x34_nissan','x34_tesla','x34_volkswagon'],'x34')
lr_top25_cols = lr_top25_cols.replace(['x35_tuesday','x35_wednesday','x35_thursday','x35_monday'],'x35')
lr_top25_cols = lr_top25_cols.replace(['x68_Aug','x68_July','x68_Mar','x68_Mar','x68_sept.'],'x68')
lr_top25_cols = lr_top25_cols.replace(['x93_asia','x93_europe'],'x93')
lr_top25_cols = lr_top25_cols['index'].unique()
lr_top25_cols = list(lr_top25_cols[:24])
cat_feats=[]
categoricals = ['x34','x35','x68','x93']
for i in categoricals:
    cat_feats.append(lr_top25_cols.index(i))
    
coef20_X_train, coef20_X_test, coef20_y_train, coef20_y_test = smote_oversampling(imp_data[lr_top25_cols], data, rand_state, categoricals=cat_feats)
cat_dict20 = {}
for i in categoricals:
    cat_dict20[i] = cat_dic[i]
    cat_dict20[i] = cat_dic[i]
coef20_X_train, coef20_X_test = dummies(coef20_X_train, coef20_X_test, cat_dict20, categoricals)
col_len_check(coef20_X_train, coef20_X_test)

# run log reg with new features
scaler_train = StandardScaler()
coef20_X_train = scaler_train.fit_transform(coef20_X_train)
scaler_test = StandardScaler()
coef20_X_test = scaler_test.fit_transform(coef20_X_test)
coef20 = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200)
coef20.fit(coef20_X_train, coef20_y_train)
coef20_y_pred = coef20.predict(coef20_X_test)
cmcoef20 = confusion_matrix(coef20_y_test, coef20_y_pred)
coef20_score = coef20.score(coef20_X_test, coef20_y_test) # score: 0.6586
coef20_prec = precision_score(coef20_y_test, coef20_y_pred) # precision: 0.3626
coef20_rec = recall_score(coef20_y_test, coef20_y_pred) # recall: 0.8422
coef20probs = coef20.predict_proba(coef20_X_test)
coef20probs = coef20probs[:, 1]
coefauc = roc_auc_score(coef20_y_test, coef20probs) # auc: 0.8133
print(f'AUC: {coefauc}')
fpr, tpr, thresholds = roc_curve(coef20_y_test, coef20probs)
plot_roc_curve(fpr, tpr)
# see which performs better

# Try using PCA components as features
# PCA
scaler_train = StandardScaler()
pca20_X_train = scaler_train.fit_transform(dummy_X_train)
scaler_test = StandardScaler()
pca20_X_test = scaler_train.fit_transform(dummy_X_test)
pca = PCA(n_components = 20)
X_train = pca.fit_transform(dummy_X_train)
X_test = pca.transform(dummy_X_test)
pca20 = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200)
explained_variance = pca.explained_variance_ratio_

logregpca_scores = cross_val_score(pca20, X_train, os_data_y, cv=kfold, scoring='roc_auc')
print('Accuracy for logreg with 20 Principal Components: ', logregpca_scores.mean()) 
# AUC: 0.9214 

# run lr with PCA
pca20 = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200)
pca20.fit(X_train, os_data_y)
pca20_y_pred = pca20.predict(X_test)
cmpca20 = confusion_matrix(y_test, pca20_y_pred)
pca20_score = pca20.score(X_test, y_test) # score: 0.8329
pca20_prec = precision_score(y_test, pca20_y_pred) # precision: 0.3626
pca20_rec = recall_score(y_test, pca20_y_pred) # recall: 0.8422
pcaprobs = pca20.predict_proba(X_test)
pcaprobs = pcaprobs[:, 1]
pcaauc = roc_auc_score(y_test, pcaprobs) # auc: 0.9007
print(f'AUC: {pcaauc}')
fpr, tpr, thresholds = roc_curve(y_test, pcaprobs)
plot_roc_curve(fpr, tpr)

# Based on accuracy score and AUC score, using PCA seems to work best for Logistic Regression
 
# Hyperparameter tuning with Grid Search - Logistic Regression
print("Running LR Grid Search")
LRclassifier = LogisticRegression()
LRpenalty = ['l1','l2']
LRC=np.logspace(0, 4, 5)
LRsolver = ['sag','saga']
LRmulti_class = ['ovr','multinomial']
LRmax_iter = [150, 200]
LRhyper = dict(C=LRC, penalty=LRpenalty, solver=LRsolver,multi_class=LRmulti_class,max_iter=LRmax_iter)

LRgrid_search = GridSearchCV(estimator = LRclassifier,
                           param_grid = LRhyper,
                           scoring = 'roc_auc',
                           cv = kfold,
                           n_jobs = -1)
LRgrid_search = LRgrid_search.fit(X_train, os_data_y)
LRbest_accuracy = LRgrid_search.best_score_ # score: 0.924
LRbest_parameters = LRgrid_search.best_params_ # {'C': 10.0, 'max_iter': 200, 'multi_class': 'multinomial', 'penalty': 'l2', 'solver': 'sag'}

# Prepare and run model on test set
cont_cols, cat_cols = cont_and_cat(test_data)
cat_data = test_data[cat_cols]
cont_data = test_data[cont_cols]
final_imp_dataLR = outliers_and_missing(cat_data, cont_data)

for col in cat_cols:
    cat_list='var'+'_'+col
    cat_list = pd.get_dummies(final_imp_dataLR[col], prefix=col, drop_first = True)
    data2=final_imp_dataLR.join(cat_list)
    final_imp_dataLR=data2
data_vars2=final_imp_dataLR.columns.values.tolist()
to_keep=[i for i in data_vars2 if i not in cat_cols]
final_imp_dataLR=final_imp_dataLR[to_keep]
col_len_check(dummy_X_train, final_imp_dataLR)
# Scale final test set
scaler_X_test = StandardScaler()
final_imp_dataLR = scaler_X_test.fit_transform(final_imp_dataLR)
# Use pca transformer to get pca's on final test set
final_X_test = pca.transform(final_imp_dataLR)

finalLR = LogisticRegression(C= 10.0, max_iter= 200, multi_class= 'multinomial', penalty= 'l2', solver= 'sag')
finalLR.fit(X_train, os_data_y)
#final_y_pred = finalXGB.predict(final_imp_data)
finalLRprobs = finalLR.predict_proba(final_X_test)
finalLRprobs = finalLRprobs[:, 1]
finalLRprobs_df = pd.DataFrame(finalLRprobs, columns=['Prob 1'])
finalLRprobs_df.to_csv(r'C:\Users\willh\fs\LogReg_probs.csv')

###########
### XGB ###
###########    
# xgb cross val
xgb_cv = Pipeline(steps=
                  [('xgboost', XGBClassifier())])
xgb_cv_scores = cross_val_score(xgb_cv, dummy_X_train, os_data_y, cv=kfold, scoring='roc_auc')
print('AUC score for XGBoost Classifier : ', xgb_cv_scores.mean())
# XGB AUC: 0.9789 

# Get feature importances of each kfold of cross validation
# xgb
feature_importances_xgb = pd.DataFrame()
xgb_cv_est = XGBClassifier()
output_xgb = cross_validate(xgb_cv_est, dummy_X_train, os_data_y, cv=kfold, scoring = 'roc_auc', return_estimator =True)
for idx,estimator in enumerate(output_xgb['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    feature_importances_xgb_temp = pd.DataFrame(estimator.feature_importances_,
                                       index = dummy_X_train.columns,
                                        columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances_xgb_temp[:15])
    feature_importances_xgb = pd.concat([feature_importances_xgb,feature_importances_xgb_temp])

# Group all the feature importances from all the kfolds and get top 25
xgb_top25_cols = feature_importances_xgb.groupby(feature_importances_xgb.index).sum().sort_values('importance', ascending=False)
xgb_top25_cols = xgb_top25_cols[:35]

xgb_top25_cols.reset_index(inplace=True)
xgb_top25_cols = xgb_top25_cols.replace(['x34_Toyota','x34_bmw','x34_chrystler','x34_ford','x34_nissan','x34_tesla','x34_volkswagon'],'x34')
xgb_top25_cols = xgb_top25_cols.replace(['x35_tuesday','x35_wednesday','x35_thursday','x35_monday'],'x35')
xgb_top25_cols = xgb_top25_cols.replace(['x68_Aug','x68_July','x68_Mar','x68_Mar','x68_sept.'],'x68')
xgb_top25_cols = xgb_top25_cols.replace(['x93_asia','x93_europe'],'x93')
xgb_top25_cols = xgb_top25_cols['index'].unique()
xgb_top25_cols = list(xgb_top25_cols[:24])
cat_feats=[]
categoricals = ['x35','x93']
for i in categoricals:
    cat_feats.append(xgb_top25_cols.index(i))
    
xgb20_X_train, xgb20_X_test, xgb20_y_train, xgb20_y_test = smote_oversampling(imp_data[xgb_top25_cols], data, rand_state, categoricals=cat_feats)
cat_dictxgb = {}
for i in categoricals:
    cat_dictxgb[i] = cat_dic[i]
    cat_dictxgb[i] = cat_dic[i]
xgb20_X_train, xgb20_X_test = dummies(xgb20_X_train, xgb20_X_test, cat_dictxgb, categoricals)
col_len_check(xgb20_X_train, xgb20_X_test)

xgb25 = XGBClassifier()
xgb25.fit(xgb20_X_train, xgb20_y_train)
xgb25_y_pred = xgb25.predict(xgb20_X_test)
cmxgb25 = confusion_matrix(xgb20_y_test, xgb25_y_pred)
xgb25_score = xgb25.score(xgb20_X_test, xgb20_y_test) # 0.8768
xgb25_prec = precision_score(xgb20_y_test, xgb25_y_pred) # 0.6776
xgb25_rec = recall_score(xgb20_y_test, xgb25_y_pred) # 0.7407
xgb25probs = xgb25.predict_proba(xgb20_X_test)
xgb25probs = xgb25probs[:, 1]
xgb25auc = roc_auc_score(xgb20_y_test, xgb25probs) # AUC: 0.9184
print(f'AUC: {xgb25auc}') 
xgb25fpr, xgb25tpr, xgb25thresholds = roc_curve(xgb20_y_test, xgb25probs)
plot_roc_curve(xgb25fpr, xgb25tpr)

# Hyperparameter tuning with Grid Search - XGB
# Grid Search XGBoost
print("Running XGB Grid Search")
XGBclassifier = XGBClassifier(objective='binary:logistic')
XGBparameters = [{'objective':['binary:logistic','reg:squarederror','multi:softmax'],'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.03, 0.05, 0.1], 'max_depth':range(2,10,1)}]
XGBgrid_search = GridSearchCV(estimator = XGBclassifier,
                           param_grid = XGBparameters,
                           scoring = 'roc_auc',
                           cv = kfold,
                           n_jobs = -1)
XGBgrid_search = XGBgrid_search.fit(xgb20_X_train, xgb20_y_train)
XGBbest_accuracy = XGBgrid_search.best_score_ # 0.9932
XGBbest_parameters = XGBgrid_search.best_params_ # best params: {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 200, 'objective': 'binary:logistic'}
print("Completed running xgb grid search")

### Prepare and run final model on test set
cont_cols, cat_cols = cont_and_cat(test_data)
cat_data = test_data[cat_cols]
cont_data = test_data[cont_cols]
final_imp_data = outliers_and_missing(cat_data, cont_data)
final_imp_data = final_imp_data[xgb_top25_cols]
for col in categoricals:
    cat_list='var'+'_'+col
    cat_list = pd.get_dummies(final_imp_data[col], prefix=col, drop_first = True)
    data1=final_imp_data.join(cat_list)
    final_imp_data=data1
data_vars=final_imp_data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_cols]
final_imp_data=final_imp_data[to_keep]
col_len_check(xgb20_X_train, final_imp_data)

finalXGB = XGBClassifier(learning_rate=0.1, max_depth=9,n_estimators= 200, objective= 'binary:logistic')
finalXGB.fit(xgb20_X_train, xgb20_y_train)
#final_y_pred = finalXGB.predict(final_imp_data)
finalxgbprobs = finalXGB.predict_proba(final_imp_data)
finalxgbprobs = finalxgbprobs[:, 1]
finalxgbprobs_df = pd.DataFrame(finalxgbprobs, columns=['Prob 1'])
finalxgbprobs_df.to_csv(r'C:\Users\willh\fs\XGB_probs.csv')


