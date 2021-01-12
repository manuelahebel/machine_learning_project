"""
Abschlussprojekt Machine Learning
Gruppe:      B
Datensatz:  Titanic
Datum:      08.12.2020

Autoren:    Manuela Hebel
            xxx
            xxx
            xxx
"""

#Impotieren alller benötigten Module/Klassen
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import timeit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
from sklearn.utils import estimator_html_repr 
#from mixed_naive_bayes import MixedNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import VotingClassifier
import joblib


#Laden des Datensatzes und Trennen in Features und Labels
path = 'full.csv'
full = pd.read_csv(path)

y = full.loc[:, 'Survived']
X = full.drop(columns = 'Survived')

#train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
Hinweis: es liegt eine Ungleichverteilung von Überlebenden/Gestorbenen vor.
Ein Test mit "stratify = y" führte aber zu schlechterer Modellqualität.
'''

#numerische Variablen
numerical_features = ['Age', 'SibSp', 'Parch', 'Sex']

#kategoriale Variablen
categorical_features = ['Pclass', 'Embarked']


######Pipeline
###Teil für Preprocessing
#Transformer für numerische Variablen
numerical_transformer = Pipeline(steps=[
    ('imputer', 'passthrough'),
    ('scaler', MinMaxScaler())])

#Transformer für kategoriale Variablen
categorical_transformer = Pipeline(steps=[                                                            
    ('imputer', SimpleImputer()),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

#ColumnTransformer führt numerical_transformer und categorical_transformer zusammen
data_transformer = ColumnTransformer(transformers=[        
    ('numerical', numerical_transformer, numerical_features),
    ('categorical', categorical_transformer, categorical_features)])

#Der Preprocessor kombiniert den ColumnTransformer
preprocessor = Pipeline(steps=[
    ('data_transformer', data_transformer), 
    ('reduce_dim', PCA())])

#Parameter-Space für Preprocessor
param_grid = {
    'preprocessor__data_transformer__numerical__imputer': [SimpleImputer(strategy = "mean"), SimpleImputer(strategy = "median"), IterativeImputer()],
    'preprocessor__data_transformer__categorical__imputer__strategy': ["constant", "most_frequent"],
    'preprocessor__reduce_dim__n_components': [5, 6]
    }


###ML-Algorithmen
##Logistische Regression
"""
Hinweis: Da sich der Vorgang für jeden Algorithmus wiederholt, ist dieser nur
für die Logistische Regression sauber kommentiert
"""
#Classifier zu Pipeline hinzufügen
classifier_LR = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))])

#Parameter-Space für Preprocessor für Algorithmus
param_grid_LR = {
    "classifier__penalty": ["l1", "l2", None],
    "classifier__solver": ["lbfgs", "liblinear", "sag", "saga", "newton-cg"],
    "classifier__C": [0.05, 0.1,0.5],
    "classifier__class_weight": ["balanced", None],
    "classifier__max_iter": [20, 25, 30]
    }

#Kombinieren der Parameter-Spaces
param_grid_LR = {**param_grid, **param_grid_LR}

#Grid-Search (getimed)
grid_search_LR = GridSearchCV(classifier_LR, param_grid=param_grid_LR, cv = 3, n_jobs = -1, verbose = 1)

start = timeit.default_timer()
grid_search_LR.fit(X_train, y_train);
end = timeit.default_timer()
runtime_LR = end - start


##SVC

classifier_SVC = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42))])

param_grid_SVC = {
    'classifier__kernel': ["linear", "poly", "rbf"],
    'classifier__probability': [False, True],
    'classifier__C': [10.0, 1.0, 0.1],
    'classifier__degree': [2, 3, 4],
    'classifier__class_weight': [None, "balanced"],
    'classifier__gamma': ["scale", "auto", 0.0001, 1]
    }

param_grid_SVC = {**param_grid, **param_grid_SVC}

grid_search_SVC = GridSearchCV(classifier_SVC, param_grid=param_grid_SVC, cv = 3, n_jobs = -1, verbose = 1)

start = timeit.default_timer()
grid_search_SVC.fit(X_train, y_train);
end = timeit.default_timer()
runtime_SVC = end - start


##Decision Tree

classifier_DT = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))])

param_grid_DT = {
    'classifier__max_depth': [3, 4, 5],
    'classifier__min_samples_leaf': [1, 3, 5],
    'classifier__criterion': ["gini", "entropy"],
    'classifier__splitter': ["best", "random"],
    'classifier__min_samples_split': [2, 3, 4],
    "classifier__max_features": [2, 4, 6],
    "classifier__min_impurity_decrease": [0, 0.1]
    }

param_grid_DT = {**param_grid, **param_grid_DT}

grid_search_DT = GridSearchCV(classifier_DT, param_grid=param_grid_DT, cv = 3, n_jobs = -1, verbose = 1)

start = timeit.default_timer()
grid_search_DT.fit(X_train, y_train);
end = timeit.default_timer()
runtime_DT = end - start


##Bayes
"""
Hinweis: ein MixedBayes ist in sklearn nicht vorhanden. Daher wurde ein anderes Modul,
dass mit sklearn kombinierbar sein soll getestet. Die Implementierung erscheint uns zweifelhaft
--> siehe Report
"""
"""
classifier_BC = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MixedNB(categorical_features=[2]))])

param_grid_BC = param_grid

grid_search_BC = GridSearchCV(classifier_BC, param_grid=param_grid_BC, cv = 3, n_jobs = -1, verbose = 1)

start = timeit.default_timer()
grid_search_BC.fit(X_train, y_train);
end = timeit.default_timer()
runtime_BC = end - start
"""


##Neuronales Netz

classifier_NN = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(random_state=42))])

param_grid_NN = {
    'classifier__hidden_layer_sizes': [(10,5),(10,10), (5,10), (5,5)],
    'classifier__activation': ["identity", 'tanh', 'relu'],
    'classifier__solver': ["lbfgs", 'sgd', 'adam'],
    'classifier__alpha': [0.0001, 0.05],
    'classifier__learning_rate': ['constant','adaptive'],
    "classifier__max_iter" : [100, 500, 2000]
    }

param_grid_NN = {**param_grid, **param_grid_NN}

grid_search_NN = GridSearchCV(classifier_NN, param_grid=param_grid_NN, cv = 3, n_jobs = -1, verbose = 1)

start = timeit.default_timer()
grid_search_NN.fit(X_train, y_train);
end = timeit.default_timer()
runtime_NN = end - start


##Ensemble Learning
##AdaBoost

classifier_AB = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(random_state=42))])

param_grid_AB = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__learning_rate': [0.1, 0.5, 1],
    'classifier__algorithm': ["SAMME", "SAMME.R"]
    }

param_grid_AB = {**param_grid, **param_grid_AB}

grid_search_AB = GridSearchCV(classifier_AB, param_grid=param_grid_AB, cv = 3, n_jobs = -1, verbose = 1)

start = timeit.default_timer()
grid_search_AB.fit(X_train, y_train);
end = timeit.default_timer()
runtime_AB = end - start


##Bagging/Pasting

classifier_BC = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', BaggingClassifier(random_state=42))])

param_grid_BC = {
    'classifier__n_estimators': [250, 500],
    'classifier__max_samples': [0.1, 0.2],
    'classifier__bootstrap': [True, False],
    "classifier__max_features": [2, 4, 6],
    "classifier__oob_score" : [True, False]
    }

param_grid_BC = {**param_grid, **param_grid_BC}

grid_search_BC = GridSearchCV(classifier_BC, param_grid=param_grid_BC, cv = 3, n_jobs = -1, verbose = 1)

start = timeit.default_timer()
grid_search_BC.fit(X_train, y_train);
end = timeit.default_timer()
runtime_BC = end - start


##SoftVoter

#Instanziierung der beiden gewählten Algorithmen
svm_clf = SVC(probability=True, random_state=42)
bc_clf = BaggingClassifier(bootstrap = True, random_state=42)

#VotingClassifier instanziieren
sv_clf = VotingClassifier(
    estimators=[('svm', svm_clf), 
                ('bc', bc_clf)],
    voting='soft',
    weights=None,
    n_jobs=-1)

#ab hier wie oben
classifier_SV = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', sv_clf)])

param_grid_SV = {
    'classifier__svm__kernel': ["linear", "poly", "rbf"],
    'classifier__svm__C': [10.0, 1.0],
    'classifier__svm__degree': [2, 3],
    'classifier__svm__class_weight': [None, "balanced"],
    'classifier__svm__gamma': ["scale", "auto"],
    'classifier__bc__n_estimators': [250, 500],
    'classifier__bc__max_samples': [0.1, 0.2],
    "classifier__bc__max_features": [2, 4, 6],
    "classifier__bc__oob_score" : [True, False]
    }

param_grid_SV = {**param_grid, **param_grid_SV}

grid_search_SV = GridSearchCV(classifier_SV, param_grid=param_grid_SV, cv = 3, n_jobs = -1, verbose = 1)

start = timeit.default_timer()
grid_search_SV.fit(X_train, y_train);
end = timeit.default_timer()
runtime_SV = end - start


#####Modellauswertung

grid_search = grid_search_SV

#Anzeige der Parameter des besten Modells
print('Best parameters found:\n', grid_search.best_params_)

#Anzeige des Scores des besten Modells
grid_search.best_score_

#Vorhersage der Testdaten und Anzeige der Akkuratheit
y_pred = grid_search.predict(X_test)

accuracy_score(y_test, y_pred)

#Konfusionsmatrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=[0,1],
            yticklabels=[0,1])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

#Klassifikations-Report
report = classification_report(y_test, y_pred)
print(report)

#Diagramm der Pipeline
set_config(display='diagram')
grid_search.best_estimator_
with open('titanic_DT_pipeline_estimator.html', 'w') as f:  
    f.write(estimator_html_repr(grid_search.best_estimator_))

#########Dumpen (Abspeichern der Modelle für zukünftige Nutzung)
joblib.dump(grid_search_LR, "model_LR.joblib")
joblib.dump(grid_search_SVC, "model_SVC.joblib")
joblib.dump(grid_search_NN, "model_NN.joblib")
joblib.dump(grid_search_DT, "model_DT.joblib")
joblib.dump(grid_search_AB, "model_AB.joblib")
joblib.dump(grid_search_BC, "model_BC.joblib")
joblib.dump(grid_search_SV, "model_SV.joblib")
