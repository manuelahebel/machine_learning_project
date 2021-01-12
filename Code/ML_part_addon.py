"""
Abschlussprojekt Machine Learning
Gruppe:      B
Datensatz:  Titanic
Datum:      08.12.2020

Autoren:    Jochen Zubrod 
            Ralf Bock
            Said Oudou 
            Manuela Hebel
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import timeit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
from sklearn.utils import estimator_html_repr 
import joblib


path = 'full_FareC_Child.csv'
full = pd.read_csv(path)

y = full.loc[:, 'Survived']
X = full.drop(columns = 'Survived')
X.dtypes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Einziger Unterschied zum Hauptteil - zwei numerische Spalten mehr
numerical_features = ['Age', 'SibSp', 'Parch', 'Sex', "FareC", "Child"]

categorical_features = ['Pclass', 'Embarked']


numerical_transformer = Pipeline(steps=[
    ('imputer', 'passthrough'),
    ('scaler', MinMaxScaler())])

categorical_transformer = Pipeline(steps=[                                                            
    ('imputer', SimpleImputer()),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

data_transformer = ColumnTransformer(transformers=[        
    ('numerical', numerical_transformer, numerical_features),
    ('categorical', categorical_transformer, categorical_features)])

preprocessor = Pipeline(steps=[
    ('data_transformer', data_transformer), 
    ('reduce_dim', PCA())])

param_grid = {
    'preprocessor__data_transformer__numerical__imputer': [SimpleImputer(strategy = "mean"), SimpleImputer(strategy = "median"), IterativeImputer()],
    'preprocessor__data_transformer__categorical__imputer__strategy': ["constant", "most_frequent"],
    'preprocessor__reduce_dim__n_components': [6, 8, 10]
    }


classifier_SVC = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(max_iter = 1000, random_state=42))])

param_grid_SVC = {
    'classifier__kernel': ["linear", "poly", "rbf"],
    'classifier__probability': [False, True],
    'classifier__C': [10.0, 1.0, 0.1],
    'classifier__degree': [2, 3, 4],
    'classifier__class_weight': [None, "balanced"],
    'classifier__gamma': ["scale", "auto", 0.0001, 1]
    }

param_grid_SVC = {**param_grid, **param_grid_SVC}

grid_search_SVC = GridSearchCV(classifier_SVC, param_grid=param_grid_SVC, cv = 3, n_jobs = -1, verbose = 2)

start = timeit.default_timer()
grid_search_SVC.fit(X_train, y_train);
end = timeit.default_timer()
runtime_SVC = end - start


grid_search = grid_search_SVC

print('Best parameters found:\n', grid_search.best_params_)

grid_search.best_score_

y_pred = grid_search.predict(X_test)

accuracy_score(y_test, y_pred)

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=[0,1],
            yticklabels=[0,1])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

report = classification_report(y_test, y_pred)
print(report)

set_config(display='diagram')
grid_search.best_estimator_
with open('titanic_SVC_new_pipeline_estimator.html', 'w') as f:  
    f.write(estimator_html_repr(grid_search.best_estimator_))

joblib.dump(grid_search_SVC, "model_SVC_new.joblib")
