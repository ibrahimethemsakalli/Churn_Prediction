import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ipykernel.pylab.config import InlineBackend
from jedi.api.refactoring import inline
from imblearn.over_sampling import SMOTE


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
from sklearn.model_selection import KFold
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

# df.loc[df["Balance"]>221532.800000,"Exited"]
df["Balance"].hist()
plt.show()

df["HasBalance"] = df["Balance"].apply(lambda x: 1 if x>0 else 0)

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
a.head()
df = pd.concat([df,a],axis=1)
df.head()

df["EstimatedSalary_Qcut"] = pd.qcut(df["EstimatedSalary"],10,labels=[1,2,3,4,5,6,7,8,9,10])
df.groupby("EstimatedSalary_Qcut").agg({"Exited":["mean","count"]})

df["Surname_Count"] = df["Surname"].apply(lambda x: len(x))


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
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)



smote = SMOTE()
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
accuracy_score(y_test, y_pred)

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


# RF

rf_model = RandomForestClassifier(random_state=12345).fit(X, y)

cross_val_score(rf_model, X, y, cv=10).mean()

rf_params = {"n_estimators": [200, 500, 1000],
             "max_features": [5, 7, 9],
             "min_samples_split": [5, 8,10],
             "max_depth": [3,5, None]}

rf_model = RandomForestClassifier(random_state=12345)

gs_cv = GridSearchCV(rf_model,
                     rf_params,
                     cv=10,
                     n_jobs=-1,
                     verbose=2).fit(X, y)

gs_cv.best_params_

rf_tuned = RandomForestClassifier(**gs_cv.best_params_)
cross_val_score(rf_tuned, X, y, cv=10).mean()




# LightGBM


lgbm = LGBMClassifier(random_state=12345)
cross_val_score(lgbm, X, y, cv=10).mean()

# model tuning
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
               "n_estimators": [500, 1000, 1500],
               "max_depth": [3, 5, 8]}

gs_cv = GridSearchCV(lgbm,
                     lgbm_params,
                     cv=5,
                     n_jobs=-1,
                     verbose=2).fit(X, y)

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X, y)
cross_val_score(lgbm_tuned, X, y, cv=10).mean()

feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()

# eda
# data prep
# feature eng.
# model
# tahmin
# model tuning
# final model
# feature importance

# TUM MODELLER CV YONTEMI (ÖDEV BUNA GORE OLACAK)

models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('SVM', SVC(gamma='auto')),
          ('XGB', GradientBoostingClassifier()),
          ("LightGBM", LGBMClassifier()),
          ("XBoost", XGBClassifier())]

# evaluate each model in turn
results = []
names = []

'''for name, model in models:
    kfold = KFold(n_splits=10, random_state=123456)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)'''

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


"""x_train.columns
lgbm = LGBMClassifier(random_state=12345)
lgbm.fit(x_train,y_train)
accuracy_score(y_test,lgbm.predict(x_test))"""




from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
tree_clf = DecisionTreeClassifier()
knn_clf= KNeighborsClassifier()
bgc_clf=BaggingClassifier()
gbc_clf=GradientBoostingClassifier()
abc_clf= AdaBoostClassifier()

voting_clf = VotingClassifier(
estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf), ('tree', tree_clf),('knn', knn_clf),('bg', bgc_clf),
            ('gbc', gbc_clf),('abc', abc_clf)],voting='hard')
voting_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
for clf in  (log_clf, rnd_clf, svm_clf,tree_clf,knn_clf,bgc_clf,gbc_clf,abc_clf,voting_clf):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

#####
lgbm = LGBMClassifier(random_state = 12345)

lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
              "n_estimators": [100, 500, 1000],
              "max_depth":[3, 5, 8]}

gs_cv = GridSearchCV(lgbm,
                     lgbm_params,
                     cv = 10,
                     n_jobs = -1,
                     verbose = 2).fit(x_train,y_train)

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(x_train,y_train)
y_pred = lgbm_tuned.predict(x_test)
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
                     verbose = 2).fit(x_train,y_train)

xgb_tuned = GradientBoostingClassifier(**xgb_cv_model.best_params_,random_state=12345)
xgb_tuned = xgb_tuned.fit(x_train,y_train)
y_pred = xgb_tuned.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
print(acc_score)

#SVM

from sklearn import svm
classifier = svm.SVC(class_weight={0:0.60, 1:0.40},random_state=12345)
svm_tuned = classifier.fit(x_train, y_train)
y_pred = svm_tuned.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
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



