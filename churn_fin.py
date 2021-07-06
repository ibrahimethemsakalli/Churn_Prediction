import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ipykernel.pylab.config import InlineBackend
from jedi.api.refactoring import inline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    classification_report

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);
pd.set_option("display.float_format", lambda x : "%5f" %x)

pd.read_csv("C:/Users/Erdal/PycharmProjects/dsmlbc/datasets/churn.csv")
def load_df():
    df = pd.read_csv("C:/Users/Erdal/PycharmProjects/dsmlbc/datasets/churn.csv",index_col=0)
    df.drop("CustomerId",axis=1,inplace=True)
    return df
df = load_df()

# Explanatory Data Analysis (EDA) & Data Visualization

df = load_df()
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()


# Categorical Varaible Analyze
cat_cols = [col for col in df.columns if len(df[col].value_counts())<12 and col not in "Exited"]
for i in cat_cols:
    sns.countplot(x=i, data=df)
    plt.show()

def cat_summary_long(data,categorical_cols ,number_of_classes=10):
    var_count = 0
    vars_more_classes = []
    for var in data:
        if var in categorical_cols:
            print(pd.DataFrame({var: data[var].value_counts(),
                                "Ratio": 100 * data[var].value_counts() / len(data)}), end="\n\n")
            var_count += 1
        else:
            vars_more_classes.append(var)
    print("{} categorical variables have been described".format(var_count), end="\n\n")
    print("There are {} variables have more than {} classes".format(len(vars_more_classes), number_of_classes),
          end="\n\n")
    print("Variables name have more than {} classes".format(number_of_classes), end="\n")
    print(vars_more_classes)

cat_summary_long(df,cat_cols,12)

# Numerical Varaible Analyze

df.describe().T
df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

num_cols = [col for col in df.columns if len(df[col].value_counts())> 12 and df[col].dtypes !="O"]

def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins =20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        sns.boxplot(x=col,data=data)
        plt.show()
        col_counter += 1
    print(col_counter,"variables have been plotted")
hist_for_nums(df,num_cols)

#df[Balance] == 0


# Target Analyze

df["Exited"].value_counts()

# Target Analysis Based on Categorical Variables
df.groupby("Gender")["Exited"].mean()

def target_summary_with_cat(data, target,cat_names):
    for var in cat_names:
        print("\t\t\t\t",var,"\n\n",pd.DataFrame({target+"_MEAN" : data.groupby(var)[target].mean(),var+"_COUNT": data[var].value_counts(),
                            var+"_RATIO": 100 * data[var].value_counts() / len(data)}), end = "\n\n\n")


target_summary_with_cat(df,"Exited",cat_cols)

# Target Analysis Based on Numeric Variables
df.groupby("Exited")["Age"].mean()

def target_summary_with_num(data, target, num_names):
    for col in num_names:
        print(pd.DataFrame({col: data.groupby(target)[col].median()}), end="\n\n\n")

target_summary_with_num(df,"Exited",num_cols)

# Analyze Numeric Varaible According To Each Other
num_cols
sns.scatterplot(x = "Balance", y = "Age", data =df)
plt.show()

sns.lmplot(x = "EstimatedSalary", y = "CreditScore", data = df)
plt.show()

df.corr()

# We will use heatmap also in Feature Engineering
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p = sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
plt.show()

# DATA PRE-PROCESSING & FEATURE ENGINEERING

# Varaible has outlier
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

df["Age"].quantile(0.99)


def has_outliers(dataframe, num_col_names, plot = False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis = None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col,":",number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x = dataframe[col])
                plt.show()
    return variable_names

outlier_list = has_outliers(df,num_cols,True)

outlier_thresholds(df,"CreditScore")
outlier_thresholds(df,"Age")

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
for i in outlier_list:
    replace_with_thresholds(df,i)
#Check has outlier
has_outliers(df,num_cols)

for i in df.columns:
    if i not in "Exited":
        print(i)
        print(df.groupby("Exited")[i].describe())

# Feature Engineering

df["CreditScore<405"] = df["CreditScore"].apply(lambda x: 1 if x< 405 else 0)
df["CreditScore<405"].value_counts()
# df.loc[df["Balance"]>221532.800000,"Exited"]
df["Balance"].hist()
plt.show()

df["HasBalance"] = df["Balance"].apply(lambda x: 1 if x>0 else 0)
df.groupby("HasBalance")["Exited"].mean()

df["NEW_NUMOFPRODUCTS"] = df["NumOfProducts"] - df["HasCrCard"]

df["Balance_1"] = df["Balance"].apply(lambda x: 1 if x == 0 else x)
df["EstimatedSal/Balance_1"] = df["EstimatedSalary"]/df["Balance_1"]
df.drop("Balance_1",axis=1,inplace=True)

df.groupby(["Geography","Gender"]).agg({"Age":"mean","Exited":"mean","Gender":"count"})
df["France_Female"] = 0
df.loc[(df["Geography"]=="France")&(df["Gender"]=="Female"),"France_Female"] = 1
df["Germany_Female"] = 0
df.loc[(df["Geography"]=="Germany")&(df["Gender"]=="Female"),"Germany_Female"] = 1
df["Spain_Female"] = 0
df.loc[(df["Geography"]=="Spain")&(df["Gender"]=="Female"),"Spain_Female"] = 1

df.groupby(["Exited","Geography"]).agg({"Age":"mean"})
a = pd.DataFrame(pd.qcut(df["Age"],4,labels=[1,2,3,4]))
a.rename(columns={"Age":"Age_qcut"},inplace=True)
a.tail()
df = pd.concat([df,a],axis=1)
df.head()

df["EstimatedSalary_Qcut"] = pd.qcut(df["EstimatedSalary"],10,labels=[1,2,3,4,5,6,7,8,9,10])
df.groupby("EstimatedSalary_Qcut").agg({"Exited":["mean","count"]})

df["Surname_Count"] = df["Surname"].apply(lambda x: len(x))

df["Exited"].value_counts()

# WELLDONE
# df.loc[df["NumOfProducts"]>3,"Exited"]

def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = dataframe.columns.tolist()   #Eski hali : list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns
cat_cols = [col for col in df.columns if len(df[col].value_counts())<12 and col not in "Exited"]
df, new_cols_ohe = one_hot_encoder(df, cat_cols, False)

cols_need_scale = [col for col in df.columns if col not in new_cols_ohe and col not in "Exited" and df[col].dtypes != "O"]

"""plt.figure(figsize=(36,30))  # on this line I just set the size of figure to 12 by 10.
p = sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
plt.show()"""





def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.25)
    quartile3 = variable.quantile(0.75)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.10)
        quartile3 = variable.quantile(0.90)
        interquantile_range = quartile3 - quartile1
        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)


for col in cols_need_scale:
    df[col] = robust_scaler(df[col])

df.drop(["Surname"],axis=1,inplace=True)


y = df["Exited"]
X = df.drop(["Exited"], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=11,stratify=y)

"""# import the Random Under Sampler object.
from imblearn.under_sampling import RandomUnderSampler

# create the object.
under_sampler = RandomUnderSampler()

# fit the object to the training data.
X_train, y_train = under_sampler.fit_resample(X_train,y_train)"""

"""# import the NearMiss object.
from imblearn.under_sampling import NearMiss

# create the object with auto
near = NearMiss(sampling_strategy="not minority")

# fit the object to the training data.
X_train, y_train = near.fit_resample(X_train, y_train)"""

"""# import the TomekLinks object.
from imblearn.under_sampling import TomekLinks

# instantiate the object with the right ratio strategy.
tomek_links = TomekLinks(sampling_strategy='majority')

# fit the object to the training data.
X_train, y_train = tomek_links.fit_resample(X_train, y_train)"""

"""# import the ClusterCentroids object.
from imblearn.under_sampling import ClusterCentroids

# instantiate the object with the right ratio.
cluster_centroids = ClusterCentroids(sampling_strategy="auto")

# fit the object to the training data.
X_train, y_train = cluster_centroids.fit_resample(X_train, y_train)"""

# import the EditedNearestNeighbours object.
from imblearn.under_sampling import EditedNearestNeighbours

# create the object to resample the majority class.
enn = EditedNearestNeighbours(sampling_strategy="majority",)

# fit the object to the training data.
X_train, y_train = enn.fit_resample(X_train, y_train)

"""# import the NeighbourhoodCleaningRule object.
from imblearn.under_sampling import NeighbourhoodCleaningRule

# create the object to resample the majority class.
ncr = NeighbourhoodCleaningRule(sampling_strategy="majority")

# fit the object to the training data.
X_train, y_train = ncr.fit_resample(X_train, y_train)"""

"""# import the Random Over Sampler object.
from imblearn.over_sampling import RandomOverSampler

# create the object.
over_sampler = RandomOverSampler()

# fit the object to the training data.
X_train, y_train = over_sampler.fit_resample(X_train, y_train)"""

"""# import the SMOTETomek
from imblearn.over_sampling import SMOTE

# create the  object with the desired sampling strategy.
smote = SMOTE(sampling_strategy='minority')

# fit the object to our training data
X_train, y_train = smote.fit_resample(X_train, y_train)"""

"""# import the ADASYN object.
from imblearn.over_sampling import ADASYN

# create the object to resample the majority class.
adasyn = ADASYN(sampling_strategy="minority")

# fit the object to the training data.
X_train, y_train = adasyn.fit_resample(X_train, y_train)"""

"""# import the SMOTETomek.
from imblearn.combine import SMOTETomek

# create the  object with the desired sampling strategy.
smotemek = SMOTETomek(sampling_strategy='auto')

# fit the object to our training data.
X_train, y_train = smotemek.fit_resample(X_train, y_train)"""

"""# import the SMOTEENN.
from imblearn.combine import SMOTEENN

# create the  object with the desired samplig strategy.
smoenn = SMOTEENN(sampling_strategy='minority')

# fit the object to our training data.
X_train, y_train = smoenn.fit_resample(X_train, y_train)"""

"""
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=11)

pipeline = imbpipeline(steps=[['RUS', ClusterCentroids(random_state=)],
                              ['classifier', LogisticRegression(random_state=11,
                                                                max_iter=1000)]])

stratified_kfold = StratifiedKFold(n_splits=3,
                                   shuffle=True,
                                   random_state=11)

param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           scoring='recall',
                           cv=stratified_kfold,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
cv_score = grid_search.best_score_
test_score = grid_search.score(X_test, y_test)
print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')

"""




#Model Tunning

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(gamma="auto")
tree_clf = DecisionTreeClassifier()
knn_clf= KNeighborsClassifier()
bgc_clf=BaggingClassifier()
gbc_clf=GradientBoostingClassifier()
abc_clf= AdaBoostClassifier()
lgbm_clf = LGBMClassifier(random_state = 12345)
nb_clf = GaussianNB()
xgb_clf = GradientBoostingClassifier(random_state=12345)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
rus = RandomUnderSampler()
nm = NearMiss()
tl = TomekLinks()
cc = ClusterCentroids()
enn = EditedNearestNeighbours()
ncr = NeighbourhoodCleaningRule()
ros = RandomOverSampler()
smt = SMOTE(random_state=11)
ada = ADASYN()
smtmk = SMOTETomek()
smtenn = SMOTEENN()

stratified_kfold = StratifiedKFold(n_splits=3,
                                   shuffle=True,
                                   random_state=11)

param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

voting_clf = VotingClassifier(
estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf), ('tree', tree_clf),('knn', knn_clf),('bg', bgc_clf),
            ('gbc', gbc_clf),('abc', abc_clf),("lgbm", lgbm_clf),("nb", nb_clf),("xgb", xgb_clf)],voting='hard')
#voting_clf.fit(X_train, y_train)


for clf in  (log_clf, rnd_clf, svm_clf,tree_clf,knn_clf,bgc_clf,gbc_clf,abc_clf,lgbm_clf,nb_clf,xgb_clf,voting_clf):
    for smpmthd in (rus, nm, tl, cc, enn, ncr, ros, smt, ada, smtmk, smtenn):

        pipeline = imbpipeline(steps=[['UnderOverSamplingMethods', smpmthd],
                                      ['classifier', clf]])

        for a in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            #cross_val_score(estimator=pipeline, X_train, y_train, scoring=a, cv=stratified_kfold, n_jobs=-1).mean()
            cv_results = cross_val_score(pipeline, X_train, y_train, cv=stratified_kfold, scoring=a)


            """grid_search = GridSearchCV(estimator=pipeline,
                                       scoring=a,
                                       param_grid=param_grid,
                                       cv=stratified_kfold,
                                       n_jobs=-1)"""

            """cv_score = grid_search.best_score_
            test_score = grid_search.score(X_test, y_test)"""
            #test_score = 1
            print(clf,smpmthd,a,f'Cross-validation score: {cv_results.mean()}')




        """clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__,"Accuracy Score :", accuracy_score(y_test, y_pred))
        print(clf.__class__.__name__,"Precision Score :", precision_score(y_test, y_pred))
        print(clf.__class__.__name__,"Recall Score :", recall_score(y_test, y_pred))
        print(clf.__class__.__name__,"F1 Score :", f1_score(y_test, y_pred))"""














"""

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler ENN
from imblearn.over_sampling import ADASYN

steps = [('over', BorderlineSMOTE()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
oversample = ADASYN()
under = RandomUnderSampler()
X_train, y_train = oversample.fit_resample(X_train, y_train)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X_test, y_test, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % mean(scores))
"""


"""
smote = BorderlineSMOTE()
X_train, y_train = smote.fit_resample(X_train,y_train)

X_train.head()
y_train.head()
y_train.value_counts()

log_model = LogisticRegression().fit(X_train, y_train)
log_model.intercept_
log_model.coef_

log_model.predict(X_test)[0:10]
y[0:10]

log_model.predict_proba(X_test)[0:10]
y_pred = log_model.predict(X_test)

confusion_matrix(y_test,y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import time
"""


""""
models = [GaussianNB(), DecisionTreeClassifier(), SVC()]
names = ["Naive Bayes", "Decision Tree", "SVM"]
for model, name in zip(models, names):
    print(name)
    start = time.time()
    for score in ["accuracy", "precision", "recall","f1"]:
        print(score," : ",cross_val_score(, X, y, scoring=score, cv=10).mean())
    print("Duration : ",time.time() - start,"sec")
"""

"""
folds = KFold(n_splits = 5, shuffle = True, random_state = 35)
scores = []

for n_fold, (train_index, valid_index) in enumerate(folds.split(X,y)):
    print('\n Fold '+ str(n_fold+1 ) + 
          ' \n\n train ids :' +  str(train_index) +
          ' \n\n validation ids :' +  str(valid_index))
    
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    acc_score = accuracy_score(y_test, y_pred)
    scores.append(acc_score)
    print('\n Accuracy score for Fold ' +str(n_fold+1) + ' --> ' + str(acc_score)+'\n')

    
print(scores)
print('Avg. accuracy score :' + str(np.mean(scores)))

"""

"""

cross_val_score(log_model, X_test, y_test, cv=10).mean()

print(classification_report(y, y_pred))



logit_roc_auc = roc_auc_score(y, log_model.predict(X))
fpr, tpr, thresholds = roc_curve(y, log_model.predict_proba(X)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
"""

#LR

LR = LogisticRegression()
LRparam_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'max_iter': list(range(100,800,100)),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
LR_search = GridSearchCV(LR, param_grid=LRparam_grid, refit = True, verbose = 3, cv=5)

# fitting the model for grid search
LR_search.fit(X_train , y_train)
LR_search.best_params_
# summarize
print('Mean Accuracy: %.3f' % LR_search.best_score_)
print('Config: %s' % LR_search.best_params_)
LR_tuned = LogisticRegression(**LR_search.best_params_).fit(X_train,y_train)
y_pred = LR_tuned.predict(X_test)
print("LR Accuracy Score : ",accuracy_score(y_test,y_pred))
print("LR Recall Score : ",recall_score(y_test,y_pred))


# RF
kfold = StratifiedKFold(n_splits=10)

rf_model = RandomForestClassifier(random_state=12345).fit(X_train, y_train)

cross_val_score(rf_model, X_train, y_train, cv=10).mean()

rf_params = {"n_estimators": [200, 500, 1000],
             "max_features": [5, 7, 9],
             "min_samples_split": [5, 8,10],
             "max_depth": [3,5, None]}

rf_model = RandomForestClassifier(random_state=12345)

rf_gs_cv = GridSearchCV(rf_model,
                     rf_params,
                     cv=kfold,
                     n_jobs=-1,
                     verbose=2).fit(X_train, y_train)

rf_gs_cv.best_params_

rf_tuned = RandomForestClassifier(**rf_gs_cv.best_params_).fit(X_train,y_train)
y_pred = rf_tuned.predict(X_test)
print("RF Accuracy Score : ",accuracy_score(y_test,y_pred))
print("RF Recall Score : ",recall_score(y_test,y_pred))

# LightGBM

kfold = StratifiedKFold(n_splits=10)
lgbm = LGBMClassifier(random_state=12345)
cross_val_score(lgbm, X_train, y_train, cv=kfold,scoring="recall").mean()

# model tuning
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
               "n_estimators": [500, 1000, 1500],
               "max_depth": [3, 5, 8],
               "num_leaves":[30]}


gs_cv = GridSearchCV(lgbm,
                     lgbm_params,
                     cv=kfold,
                     n_jobs=1,
                     verbose=2).fit(X_train, y_train)

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
print("LGBM Accuracy Score : ",accuracy_score(y_test,y_pred))
print("LGBM Recall Score : ",recall_score(y_test,y_pred))

feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()

#####
xgb = GradientBoostingClassifier(random_state=12345)

xgb_params = {"n_estimators": [100, 500, 1000],
              "subsample" : [0.6, 0.8, 1.0],
              "max_depth" : [3, 4, 5],
              "learning_rate" : [0.1, 0.01, 0.05,],
              "min_samples_split" : [2, 5, 10]}

xgb_cv_model = GridSearchCV(xgb,
                     xgb_params,
                     cv = 5,
                     n_jobs = -1,
                     verbose = 2).fit(X_train,y_train)

xgb_tuned = GradientBoostingClassifier(**xgb_cv_model.best_params_,random_state=12345).fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
print("XGB Accuracy Score : ",accuracy_score(y_test,y_pred))
print("XGB Recall Score : ",recall_score(y_test,y_pred))


"""# eda
# data prep
# feature eng.
# model
# tahmin
# model tuning
# final model
# feature importance

# TUM MODELLER CV YONTEMI

models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('SVM', SVC(gamma='auto')),
          ('XGB', GradientBoostingClassifier()),
          ("LightGBM", LGBMClassifier()),
          ("Naive Bayes",GaussianNB()),]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123456, shuffle=True)
    cv_results = cross_val_score(model, X_test, y_test, cv=10, scoring=["accuracy","precision","recall","f1"])
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()





from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from imblearn.combine import SMOTETomek

smk = SMOTETomek()
x_train, y_train = smk.fit_resample(x_train, y_train)

x_test, y_test = smk.fit_resample(x_test, y_test)


""""""
x_train.columns
lgbm = LGBMClassifier(random_state=12345)
lgbm.fit(x_train,y_train)
accuracy_score(y_test,lgbm.predict(x_test))"""
"""




from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(gamma="auto")
tree_clf = DecisionTreeClassifier()
knn_clf= KNeighborsClassifier()
bgc_clf=BaggingClassifier()
gbc_clf=GradientBoostingClassifier()
abc_clf= AdaBoostClassifier()
lgbm_clf = LGBMClassifier(random_state = 12345)
nb_clf = GaussianNB()



voting_clf = VotingClassifier(
estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf), ('tree', tree_clf),('knn', knn_clf),('bg', bgc_clf),
            ('gbc', gbc_clf),('abc', abc_clf),("lgbm", lgbm_clf),("nb", nb_clf)],voting='hard')
voting_clf.fit(X_train, y_train)


for clf in  (log_clf, rnd_clf, svm_clf,tree_clf,knn_clf,bgc_clf,gbc_clf,abc_clf,lgbm_clf,nb_clf,voting_clf):

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__,"Accuracy Score :", accuracy_score(y_test, y_pred))
    print(clf.__class__.__name__,"Precision Score :", precision_score(y_test, y_pred))
    print(clf.__class__.__name__,"Recall Score :", recall_score(y_test, y_pred))
    print(clf.__class__.__name__,"F1 Score :", f1_score(y_test, y_pred))





#####
lgbm = LGBMClassifier(random_state = 12345)

lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
              "n_estimators": [100, 500, 1000],
              "max_depth":[3, 5, 8]}

gs_cv = GridSearchCV(lgbm,
                     lgbm_params,
                     cv = 10,
                     n_jobs = -1,
                     verbose = 2).fit(X_train,y_train)

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X_train,y_train)
y_pred = lgbm_tuned.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
print(acc_score)
#####
xgb = GradientBoostingClassifier(random_state=12345)

xgb_params = {"n_estimators": [100, 500, 1000],
              "subsample" : [0.6, 0.8, 1.0],
              "max_depth" : [3, 4, 5],
              "learning_rate" : [0.1, 0.01, 0.05,],
              "min_samples_split" : [2, 5, 10]}

xgb_cv_model = GridSearchCV(xgb,
                     xgb_params,
                     cv = 5,
                     n_jobs = -1,
                     verbose = 2).fit(X_train,y_train)

xgb_tuned = GradientBoostingClassifier(**xgb_cv_model.best_params_,random_state=12345)
xgb_tuned = xgb_tuned.fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
recall_score(y_test,y_pred)
print(acc_score)

#SVM

from sklearn import svm
classifier = svm.SVC(class_weight={0:0.60, 1:0.40},random_state=12345)
svm_tuned = classifier.fit(X_train, y_train)
y_pred = svm_tuned.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
recall_score(y_test,y_pred)
print(acc_score)

#SVM
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
#Libraries to Build Ensemble Model : Random Forest Classifier
# Create the parameter grid based on the results of random search
params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# Performing CV to tune parameters for best SVM fit
svm_model = GridSearchCV(SVC(), params_grid, cv=5)
svm_model.fit(x_train, y_train)

# View the accuracy score
print('Best score for training data:', svm_model.best_score_,"\n")

# View the best parameters for the model found using grid search
print('Best C:',svm_model.best_estimator_.C,"\n")
print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

final_model = svm_model.best_estimator_
Y_pred = final_model.predict(x_test)
Y_pred_label = list(Y_pred)
# Making the Confusion Matrix
#print(pd.crosstab(Y_test_label, Y_pred_label, rownames=['Actual Activity'], colnames=['Predicted Activity']))
print(confusion_matrix(y_test,Y_pred_label))
print("\n")
print(classification_report(y_test,Y_pred_label))

print("Training set score for SVM: %f" % final_model.score(x_train , y_train))
print("Testing  set score for SVM: %f" % final_model.score(x_test  , y_test ))

svm_model.score
"""


