import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Load Data ----------------------------------------------------------------------------------------
df = pd.read_csv('question2.csv')
df.head(5)

# 2. checing missing and null values
df.isna().sum().sum()

df.isnull().sum().sum()

# 3. Convert to dummy variables
df = pd.get_dummies(df, drop_first=True)

# 4. cleaning and tidy column names
df.columns = [col.replace('-','_').replace(' ','').replace('&','_').replace('(','_').replace(')','_') for col in df.columns]
#df.info()

#----------------------------------------------------------------------------------------------------
# 5. Split the dataset with 25% as the test set with random state 42
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Defining feature and label-------------------------------------------------------------------------
X = df.drop(['Salary_50K_yes'], axis=1)
y = df['Salary_50K_yes']

# 6. Use five-fold cross validation and pipeline to choose the best model using ALL combinations------
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define the pipeline ------------------------------------------------------------------------------
pipe = Pipeline([
    ('selector', VarianceThreshold()), 
    ('scaler', StandardScaler()),    
    ('estimator', LogisticRegression())
    
])

# Define the parameter grid for GridSearchCV--------------------------------------------------------
param_grid = [
    {
        'selector':[VarianceThreshold(threshold=0.25), 
                    SelectFromModel(
                        estimator=RandomForestClassifier(n_estimators = 10, max_depth = 5))],
        'scaler': [StandardScaler(), MinMaxScaler()],
        'estimator': [LogisticRegression()],
        'estimator__random_state': [42]
    },
    {
        'selector':[VarianceThreshold(threshold=0.25), 
                    SelectFromModel(
                        estimator=RandomForestClassifier(n_estimators = 10, max_depth = 5))],
        'scaler': [StandardScaler(), MinMaxScaler()],
        'estimator': [MLPClassifier()],
        'estimator__hidden_layer_sizes': [(10, 5)],
        'estimator__activation':['identity', 'logistic', 'tanh', 'relu'],
        'estimator__learning_rate_init': [0.01],
        'estimator__max_iter': [100],
        'estimator__random_state': [42]
        
    },
    {
        'selector':[VarianceThreshold(threshold=0.25), 
                    SelectFromModel(
                        estimator=RandomForestClassifier(n_estimators = 10, max_depth = 5))],
        'scaler': [StandardScaler(), MinMaxScaler()],
        'estimator': [RandomForestClassifier()],
        'estimator__n_estimators': [10],
        'estimator__max_depth': [5],
        'estimator__random_state': [42]
    },
    {
        'selector':[VarianceThreshold(threshold=0.25), 
                    SelectFromModel(
                        estimator=RandomForestClassifier(n_estimators = 10, max_depth = 5)
                    )],
        'scaler': [StandardScaler(), MinMaxScaler()],
        'estimator': [AdaBoostClassifier()],
        'estimator__learning_rate': [0.1],
        'estimator__random_state': [42]
    }
]

# Perform the grid search using GridSearchCV--------------------------------------------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=kfold)
grid_search.fit(X, y)

# Grid scores for all the models based on CV--------------------------------------------------------
print("Grid scores for all the models based on CV:\n")
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

# Best model ---------------------------------------------------------------------------------------
print("\nBest parameters set found on development set:", grid_search.best_params_)
print("\nBest model validation accuracy:", grid_search.best_score_)

# 7. Make prediction in the test set ----------------------------------------------------------------
gs_best = grid_search.best_estimator_
#X Test data
y_test_pred = gs_best.predict(X_test)
print('----------------------------------------------------------------')
print('\nTuned Model Stats :')
print(classification_report(y_test, y_test_pred, target_names=['class 0', 'class 1']))

# Calculate the accuracy of the classifier
print('----------------------------------------------------------------')
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy:", accuracy)

# Compute and print the confusion matrix
print('----------------------------------------------------------------')
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:\n", conf_matrix)

# Plot the confusion matrix -----------------------------------------------------------------------
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                conf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     conf_matrix.flatten()/np.sum(conf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

fig, ax = plt.subplots(figsize=(6,4), dpi=100)
sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
ax.set_title("Confusion Matrix for people with certain profile will have a salary > 50K or not", fontsize=14, pad=20)
plt.show()

# The best from the grid search is the MLPClassifier with the following parameters:----------------------------
rf = SelectFromModel(estimator = RandomForestClassifier(n_estimators = 10, max_depth = 5)).fit(X, y)
X_rf = X.iloc[:, rf.get_support(indices=True)]
X_rf.info()

X_train, X_test, y_train, y_test = train_test_split(X_rf, y, test_size=0.25, random_state = 42)

pipe = Pipeline([
    ('scaler', StandardScaler()),    
    ('estimator', MLPClassifier())
    
])

param_grid = [
    {
        'scaler': [StandardScaler()],
        'estimator': [MLPClassifier()],
        'estimator__hidden_layer_sizes': [(10, 5)],
        'estimator__activation':['identity', 'logistic', 'tanh', 'relu'],
        'estimator__learning_rate_init': [0.01],
        'estimator__max_iter': [100],
        'estimator__random_state': [42]
    }
]


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# Perform the grid search using GridSearchCV
mlp_gs = GridSearchCV(pipe, param_grid=param_grid, cv=kfold)
mlp_gs.fit(X_rf, y)

# Grid scores for MLPClassifer model based on CV--------------------------------------------------------
print("\nBest parameters set found on development set:", mlp_gs.best_params_)
print("\nBest model validation accuracy:", mlp_gs.best_score_)

#Roc curve for MLPClassifier------------------------------------------------------------------------
from sklearn.metrics import roc_curve, roc_auc_score, auc

mlp_clf = mlp_gs.best_estimator_
y_pred_prob = mlp_clf.predict_proba(X_test)[:, 1]

logit_roc_auc = roc_auc_score(y_test, mlp_clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, mlp_clf.predict_proba(X_test)[:,1])
auc = round(metrics.roc_auc_score(y_test, y_pred_prob), 4)
#plt.plot(fpr,tpr,label="MLPClassifier, AUC="+str(auc), lw=2, color='purple')
plt.figure()
plt.plot(fpr, tpr, label="MLPCLassifier, AUC="+str(auc), lw=2, color='purple')
plt.plot([0, 1], [0, 1],'r--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

## Class Membership Prediction for the test set
X_selected = X_rf
y_prediction = mlp_clf.predict(X_test)

test_data = pd.concat([X_test, y_test], axis=1)

print(y_prediction)
#append to test data
test_data['y_pred'] = y_prediction

## Predicted Probabilities for the test set
pred_prob = mlp_clf.predict_proba(X_test)

#Append the second column (i.e., corresponds to ">50K_Yes") of predicted probabilities to test set
#Note: First column corresponds to "Attrition_No"
test_data["pred_prob"] = mlp_clf.predict_proba(X_test)[:,1]

# Predicting the Likelihood that a specific employee will earn salary >50K
#save to CSV
test_data.to_csv("test_data_pred.csv")

# describe the data
salary_df = pd.DataFrame(X_selected.describe())
salary_df

#Using mean values of X_selected
print(salary_df.loc['mean',:].values)
salary_df = [salary_df.loc['mean',:].values]

#predict class membership
y_pred_emp = mlp_clf.predict(salary_df)

#Predicted Probability
pred_prob_emp = mlp_clf.predict_proba(salary_df)

#define function to display class label
def display_classLabel(y_pred_):
    label =''
    if y_pred_ ==0:
        label ='"< 50K salary"'
    else:
        label = '">50K salary"'
    print('Predicted class membership for the employee is',y_pred_,', which means',label)
    
#display predicted class label
display_classLabel(y_pred_emp)

#define function to display predicted probability
def display_PredProb(y_pred_, pred_prob):
    prob_label = ''
    prob =0
    if y_pred_==0:
        prob_label ='", which mean the probability of <50K salary is"'
        prob = pred_prob[:,0]
    else:
        prob_label ='", which means, probability of >50K salary is"'
        prob = pred_prob[:,1]
    print('Predicted probability for the salary is',prob, prob_label, prob)
    
#display predicted probability
display_PredProb(y_pred_emp, pred_prob_emp)