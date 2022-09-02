# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 01:15:17 2022

@author: Kerillos
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier 
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import mutual_info_regression
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error,confusion_matrix, accuracy_score, precision_score, recall_score,classification_report,mean_absolute_error,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.model_selection import GridSearchCV

sns.set()
warnings.filterwarnings('ignore')

def delete_columns(data):
    un_used_columns = ['region','city','id','Unnamed: 0.1','entity_type', 'entity_id','parent_id','created_by','created_at','updated_at','domain','homepage_url', 'twitter_username','logo_url','logo_width','logo_height','short_description','description', 'overview','tag_list', 'name', 'normalized_name', 'permalink', 'invested_companies']
    data.drop(columns=un_used_columns,inplace=True)
    return data
def delete_duplicates(data):
    print("Number of duplicates values is : ",data.duplicated().sum())
    data.drop_duplicates(inplace=True)
    return data
def Null_values_drop_98 (data):
    Sum = data.isnull().sum()
    percentage = (data.isnull().sum() * 100) / len(data)
    A_1 = pd.DataFrame({'SUM':Sum , "Percentage" :percentage} , index=data.columns)
    A_1['Status'] = A_1['Percentage'].apply(lambda x : 'Drop'if x > 98 else "stay")
    print(A_1)
    Columns_With_More_Than_98_OfNull = list(A_1[A_1['Status']== 'Drop'].index)
    data.drop(Columns_With_More_Than_98_OfNull , inplace=True,axis=1)
    print('Number of rows : {}\nNumber of Columns :{}'.format(data.shape[0],data.shape[1]))
    return data
def Null_values_categories(data):
    categories_columns = ['status','country_code','founded_at','category_code','state_code']
    data[categories_columns].dropna( axis='columns' ,inplace=True)
    Sum = data.isnull().sum()
    percentage = (data.isnull().sum() * 100) / len(data)
    A_2 = pd.DataFrame({'SUM':Sum , "Percentage" :percentage} , index=data.columns)
    print(A_2)
    return data
def iqr(data,col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = np.subtract(Q3,Q1)
    return Q1,Q3,IQR
def UpLower(Q1,Q3,IQR):
    upper = Q1 - (1.5 * IQR)
    lower = Q3 + (1.5*IQR)
    return upper , lower
def drop_outliers(data):
    Q1_total , Q3_total , IQR_total = iqr(data=data,col='funding_total_usd')
    Q1_rounds , Q3_rounds , IQR_rounds =iqr(data=data,col='funding_rounds')
    total_usd = pd.DataFrame({'IQR':IQR_total , 'Q1':Q1_total , 'Q3':Q3_total},index=[0])
    rounds = pd.DataFrame({'IQR':IQR_rounds , 'Q1':Q1_rounds , 'Q3':Q3_rounds},index=[1])
    Result = pd.concat([total_usd,rounds] , axis='rows')
    # For funding_total_usd
    up_total , lower_total = UpLower(Q1_total , Q3_total , IQR_total) 
    # For funding_rounds
    up_rounds , lower_rounds = UpLower(Q1_rounds , Q3_rounds , IQR_rounds)
    Result['Upper'] = [up_total,up_rounds]
    Result['Lower'] = [lower_total,lower_rounds]
    outlier_total = data[(data['funding_total_usd'] >= up_total) | (data['funding_total_usd'] <= lower_total)]
    outlier_rounds = data[(data['funding_rounds'] >= up_rounds) | (data['funding_rounds'] <= lower_rounds)]
    print(Result)
    #Remove
    data = data[(data['funding_total_usd'] >= up_total) | (data['funding_total_usd'] <= lower_total)] 
    data = data[(data['funding_rounds'] >= up_rounds) | (data['funding_rounds'] <= lower_rounds)]
    return data
def Transformation_dates(df):
    df['founded_at'] = pd.to_datetime(df['founded_at'] , format='%Y-%m-%d')
    df['founded_at'] = df['founded_at'].dt.year
    #Type your code here!
    df['closed_at'] = pd.to_datetime(df['closed_at'] , format='%Y-%m-%d' , errors='coerce')
    df['closed_at'] = df['closed_at'].dt.year

    df['first_funding_at'] = pd.to_datetime(df['first_funding_at'] , format='%Y-%m-%d', errors='coerce')
    df['first_funding_at'] = df['first_funding_at'].dt.year

    df['last_funding_at'] = pd.to_datetime(df['last_funding_at'] , format='%Y-%m-%d', errors='coerce')
    df['last_funding_at'] = df['last_funding_at'].dt.year

    df['first_milestone_at'] = pd.to_datetime(df['first_milestone_at'] , format='%Y-%m-%d', errors='coerce')
    df['first_milestone_at'] = df['first_milestone_at'].dt.year

    df['last_milestone_at'] = pd.to_datetime(df['last_milestone_at'] , format='%Y-%m-%d', errors='coerce')
    df['last_milestone_at'] = df['last_milestone_at'].dt.year
    return df
def one_hot_encode(df):
    list_ofcategory_code = df['category_code'].value_counts().index[:15]
    df['category_code'] = df['category_code'].apply(lambda x : x if x in list_ofcategory_code else 'other' )
    df_one_Hot_Encoding_category_code = pd.get_dummies(data=df['category_code'] )
    df = pd.concat([df,df_one_Hot_Encoding_category_code], axis='columns')
    df.drop('category_code',axis=1,inplace=True)
    list_ofcategory_code = df['country_code'].value_counts().index[:10]
    df['country_code'] = df['country_code'].apply(lambda x : x if x in list_ofcategory_code else 'other' )
    df_one_Hot_Encoding_country_code = pd.get_dummies(data=df['country_code'] )
    df = pd.concat([df,df_one_Hot_Encoding_country_code], axis='columns')
    df.drop('country_code',axis=1,inplace=True)
    df['status'] , _ = df['status'].factorize()
    df['state_code'] , _ = df['state_code'].factorize()
    return df , df_one_Hot_Encoding_category_code , df_one_Hot_Encoding_country_code 
def new_features(df):
    df['closed_at'] = df['status'].apply(lambda x:2021 if x in ['operating','ipo'] else 0)
    df['Age Day'] =(df['closed_at'] - df['founded_at']) *365
    df['Age Day'] = df['Age Day'].apply(lambda x :0 if x<0 else x)
    df.drop('closed_at',axis=1,inplace=True)
    return df
def remove_Null_Numerical_values(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    Numerical_Columns = [col for col in df.select_dtypes(include=numerics).columns]
    #print(f"Count of Numerical_Columns : {len(Numerical_Columns)}")
    from sklearn.impute import SimpleImputer
    imputer =SimpleImputer(strategy='mean')
    for col in Numerical_Columns:
        df[col] = imputer.fit_transform(df[[col]])
    df.drop(['last_investment_at','first_investment_at'],axis=1,inplace=True)
    print(df[Numerical_Columns].isnull().sum())
    return df
def correlation(df):
    columns_target = df.select_dtypes(['float64','object'])
    uint8_columns = df.select_dtypes(['uint8']).columns
    uint8_columns = uint8_columns.tolist()
    
    df_1 = df.copy()
    df_1 = df[columns_target.columns]
    
    plt.figure(figsize=(30,30))
    plt.title('Pearson Correlation of Features', size = 15)
    colormap = sns.diverging_palette(10, 220, as_cmap = True)
    sns.heatmap(df_1.corr(),
                cmap = colormap,
                square = True,
                annot = True,
                linewidths=0.1,vmax=1.0, linecolor='white',
                annot_kws={'fontsize':12 })
    plt.show()
    
    
def PermutationImportance_show(data,target_name):
    x = data.drop(target_name,axis=1)
    y = data[target_name]
    # making xgb model
    x_train , x_test , y_train , y_test =train_test_split(x,y,test_size=0.2,stratify=y,shuffle=True)
    my_model = XGBClassifier().fit(x_train,y_train)
    
    
    
    perm = PermutationImportance(my_model , random_state=1).fit(x_test,y_test)
    eli5.show_weights(perm , feature_names=x_test.columns.tolist())
    
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_float_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    print(mi_scores)
    plot_mi_scores(mi_scores)
    return mi_scores
def features_importance_Tree(x,y):
    ex_tree_model = ExtraTreesClassifier().fit(x,y)
    features_importance = pd.Series(ex_tree_model.feature_importances_ , index=x.columns)
    features_importance.nlargest(14).plot(kind='barh')
    
def Handling_imbalanced_data(x,y):
    smote = SMOTE()
    x , y = smote.fit_resample(x,y)
    return x ,y
def split(x,y):
    x_train ,x_test,y_train,y_test = train_test_split(x,y,stratify=y,test_size=0.2)
    return x_train ,x_test,y_train,y_test
def split_scale(x,y):
    x_train ,x_test,y_train,y_test = split(x,y)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train ,x_test,y_train,y_test
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(2)
        
    else:
        cm=cm
        

    

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def metric(y_test,y_pred,x_train,y_train,model):
    test_acc = accuracy_score(y_test,y_pred)
    train_acc = model.score(x_train,y_train)
    precision=precision_score(y_test,y_pred,average='macro')
    recall =recall_score(y_test,y_pred,average='macro')
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    F1_score = f1_score(y_test,y_pred,average='macro')
    print('test accuracy score : ',test_acc,"%")
    print('train accuracy score : ',train_acc,"%")
    print('precision score : ',precision,"%")
    print('recall score : ',recall,"%")
    print('Mean Squared Error : ',mse,"%")
    print('Mean Absolute Error : ',mae,"%")
    print('F_1 score : ',f1_score,"%")
    
    print("classification report")
    print(classification_report(y_test,y_pred))
    print("Confusion Matrix")
    conf_mat = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    fig1 = plt.figure(figsize=(7,6))
    plot_confusion_matrix(conf_mat, classes=y_train.value_counts().index.tolist(), title='Confusion matrix')
    return test_acc,train_acc,precision,recall,mse,mae,F1_score
def features_importance_rf(x_train,y_train):
    rf = RandomForestClassifier().fit(x_train,y_train)
    importances = rf.feature_importances_.tolist()
    feature_list = x_train.columns.tolist()
    features_importances = [(feature , round(importance,2))for feature , importance in zip(feature_list,importances)]
    features_importances = sorted(features_importances,key=lambda x : x[1] , reverse=True)#
    #[print('Feature : {:20}Importances : {}'.format(*n)) for n in features_importances]
    for n in features_importances:
        print('Feature : {:20}Importances : {}'.format(*n))
def Random_Search_random_forest(x_train,y_train,model):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)
    rf_randomCV = RandomizedSearchCV(estimator=model , param_distributions=random_grid,
                                n_iter=5 ,n_jobs=-1,
                                cv=3,verbose=2,random_state=42,
                                return_train_score=True)
    rf_randomCV.fit(x_train,y_train) 
    #print('Best Estimator : ',rf_randomCV.best_estimator_)
    #print('Best Parameter : ',rf_randomCV.best_params_) 
    return rf_randomCV
                           
def grid_search(x_train,y_train,model):
    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
     } 
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 3, verbose = 2, return_train_score=True,n_jobs=-1)
    grid_search.fit(x_train, y_train)
    pprint('Best Estimator : ',grid_search.best_estimator_)
    pprint('Best Parameter : ',grid_search.best_params_) 
    return grid_search

                               