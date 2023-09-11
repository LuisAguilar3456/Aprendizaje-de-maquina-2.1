#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('drug200.csv')
print(df.columns)
print("[",df.shape[0],df.shape[1],"]")
print(df.head())


# In[3]:


plt.figure(figsize=[10,10])

plt.subplot(3,2,1)
plt.hist(df['Age'])
plt.title('Age')
plt.ylabel("Frecuencia")

plt.subplot(3,2,2)
plt.bar(df["Sex"].value_counts().index,df['Sex'].value_counts())
plt.title("Sex")
plt.ylabel("Frecuencia")

plt.subplot(3,2,3)
plt.bar(df["BP"].value_counts().index,df['BP'].value_counts())
plt.title("BP")
plt.ylabel("Frecuencia")

plt.subplot(3,2,4)
plt.bar(df['Cholesterol'].value_counts().index,df['Cholesterol'].value_counts())
plt.title('Cholesterol')
plt.ylabel("Frecuencia")

plt.subplot(3,2,5)
plt.hist(df['Na_to_K'])
plt.title('Na_to_K')
plt.ylabel("Frecuencia")

plt.subplot(3,2,6)
plt.bar(df['Drug'].value_counts().index,df['Drug'].value_counts())
plt.title('Drug')
plt.ylabel("Frecuencia")


# In[4]:


df=pd.get_dummies(df, columns=["Sex",'BP', 'Cholesterol'])
print(df.head())


# In[5]:


df=df[['Age','Sex_F', 'Sex_M','BP_HIGH', 'BP_LOW','BP_NORMAL','Cholesterol_HIGH', 'Cholesterol_NORMAL','Na_to_K','Drug']]
print(df.shape)
df.head()


# # SELECT THE FEATURES AND THE TARGET 

# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[7]:


X=df.drop("Drug",axis=1)
y=df['Drug']
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=42)
print("Training Set: [{},{}]".format(X_train.shape[0],X_train.shape[1]))
print("Test Set: [{},{}]".format(X_test.shape[0],X_test.shape[1]))


# # REGRESIÓN LOGISSTICA MULTICLASE

# In[8]:


Predicciones={"DrugY":0,"drugA":1,"drugB":2,"drugC":3,"drugX":4}
print(Predicciones)


# In[67]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, multi_class='ovr',solver="saga")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión de Regresión Logistica Multiclase: {accuracy * 100:.2f}%')


# In[68]:


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión Regresión Logistica Multiclase')
plt.show()


# # Support Vector Machines

# In[69]:


from sklearn.svm import SVC
svm_classifier = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo SVM: {accuracy * 100:.2f}%")


# In[70]:


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión SVM Multiclase')
plt.show()


# # Random Forest

# In[71]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=1, random_state=42)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy*100:.2f} %")


# In[72]:


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión Random Forest Multiclase')
plt.show()


# # K-Nearest Neighbors (K-NN)

# In[73]:


from sklearn.neighbors import KNeighborsClassifier
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {} %".format(accuracy*100))


# In[74]:


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión K-Nearest Neighbors Multiclase')
plt.show()


# # UTILIZAR RANDOM SEARCH

# ## LOGISTIC REGRESSION

# In[75]:


from sklearn.model_selection import cross_val_score

def evaluate_model(C, solver, multi_class):
    model = LogisticRegression(C=C, solver=solver, multi_class=multi_class, random_state=42,max_iter=10000)
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    bias = 1 - np.mean(accuracy_scores)
    variance = np.var(accuracy_scores, ddof=1)
    
    return bias, variance, 1 - np.mean(accuracy_scores),accuracy_scores

bias, variance, error ,accuracy= evaluate_model(C=1.0, solver='lbfgs', multi_class='multinomial')
print(f"Bias: {bias:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Error: {error:.2f}")
print("Accuracy: {}%".format(np.mean(accuracy)*100))


# In[76]:


#LEARNING CURVE DE UN MODELO DE REGRESIÓN LOGISTICA SIMPLE CON LOS PRIMEROS HIPERPARAMETROS
from sklearn.model_selection import learning_curve
model = LogisticRegression(C=1.0,solver='lbfgs',multi_class="multinomial",max_iter=10000)
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=train_sizes, cv=10, scoring='accuracy'
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.title("Learning Curve for Logistic Regression")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Accuracy")
plt.legend(loc="best")
plt.show()


# In[77]:


#HACER UN ANALISIS DE VARIANZA, BIAS Y ACCURACY MIENTRAS DE MODIFICAN HIPERPARAMETROS COMO EL SOLVER Y EL C
bias_lfg=[]
variance_lfg=[]
error_lfg=[]
accuracy_lfg=[]
for c in np.round(np.arange(0.1,3.1,0.1),2):
    bias, variance, error ,accuracy= evaluate_model(C=c, solver='lbfgs', multi_class='multinomial')
    bias_lfg.append(bias)
    variance_lfg.append(variance)
    error_lfg.append(error)
    accuracy_lfg.append(np.mean(accuracy))

bias_newton=[]
variance_newton=[]
error_newton=[]
accuracy_newton=[]   

for c in np.round(np.arange(0.1,3.1,0.1),2):
    bias, variance, error ,accuracy= evaluate_model(C=c, solver='newton-cg', multi_class='multinomial')
    bias_newton.append(bias)
    variance_newton.append(variance)
    error_newton.append(error)
    accuracy_newton.append(np.mean(accuracy))
    
bias_sag=[]
variance_sag=[]
error_sag=[]
accuracy_sag=[]   

for c in np.round(np.arange(0.1,3.1,0.1),2):
    bias, variance, error ,accuracy= evaluate_model(C=c, solver='sag', multi_class='multinomial')
    bias_sag.append(bias)
    variance_sag.append(variance)
    error_sag.append(error)
    accuracy_sag.append(np.mean(accuracy))


# In[78]:


plt.figure(figsize=[20,12])

plt.subplot(2,2,1)
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),bias_lfg,label="lbfgs",marker='*')
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),bias_newton,label="newton-cg")
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),bias_sag,label="sag")
plt.xlabel("C: Parametro de regularización")
plt.ylabel ("Bias")
plt.title("C vs bias")
plt.legend()


plt.subplot(2,2,2)
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),variance_lfg,label="lbfgs",marker='*')
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),variance_newton,label="newton-cg")
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),variance_sag,label="sag")
plt.xlabel("C: Parametro de regularización")
plt.ylabel ("Variance")
plt.title("Variance vs bias")
plt.legend()


plt.subplot(2,2,3)
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),error_lfg, label="lbfgs", marker='*')
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),error_newton,label="newton-cg")
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),error_sag,label="sag")
plt.xlabel("C: Parametro de regularización")
plt.ylabel ("Error")
plt.title("Error vs bias")
plt.legend()


plt.subplot(2,2,4)
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),accuracy_lfg, label="lbfgs",marker='*')
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),accuracy_newton,label="newton-cg")
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),accuracy_sag,label="sag")
plt.xlabel("C: Parametro de regularización")
plt.ylabel ("Accuracy")
plt.title("Accuracy vs bias")

plt.grid(False)
plt.legend()
plt.show()


# In[79]:


from sklearn.model_selection import RandomizedSearchCV
param_dist = {
    'C': np.logspace(-3, 3, 7),  # Rango de valores para C (regularización)
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Solucionadores posibles
    'multi_class': ['ovr', 'multinomial']  # Tipo de clasificación multiclase
}
base_model = LogisticRegression(random_state=42,max_iter=100000)
random_search = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print("Mejores hiperparámetros encontrados:")
print(best_params)


# In[80]:


#Mejores hiperparámetros encontrados:
#{'solver': 'newton-cg', 'multi_class': 'ovr', 'C': 10.0}
model=LogisticRegression(solver="newton-cg",multi_class='ovr',C=10.0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy de Logistic Regresión con los hiperparametros encontrados por el Random Search")
print("{} %".format(accuracy*100))


# In[81]:


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión Regresión Logistica Multiclas')
plt.show()


# In[82]:


yTrain_pred=model.predict(X_train)
accuracy=accuracy_score(y_train, yTrain_pred)
print("Accuracy Training DataSet: {}%".format(accuracy*100))

cm = confusion_matrix(y_train, yTrain_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión Regresión Logistica Multiclas')
plt.show()


# In[84]:


model = LogisticRegression(solver="newton-cg",multi_class='ovr',C=10.0)
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=train_sizes, cv=10, scoring='accuracy'
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.title("Learning Curve for Logistic Regression - Best Model")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Accuracy")
plt.legend(loc="best")
plt.show()


# ## SUPPORT VECTOR MACHINES

# In[85]:


def evaluate_model(C, solver):
    model = SVC(kernel=solver, C=C, random_state=42, max_iter=10000)
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    bias = 1 - np.mean(accuracy_scores)
    variance = np.var(accuracy_scores, ddof=1)
    
    return bias, variance, 1 - np.mean(accuracy_scores),accuracy_scores

bias, variance, error ,accuracy= evaluate_model(C=1.0, solver='poly')
print(f"Bias: {bias:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Error: {error:.2f}")
print("Accuracy: {}%".format(np.mean(accuracy)*100))


# In[86]:


model = SVC(kernel='poly', C=1.0, random_state=42, max_iter=10000)
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=train_sizes, cv=10, scoring='accuracy'
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.title("Learning Curve for Support Vector Machines")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Accuracy")
plt.legend(loc="best")
plt.show()


# In[87]:


bias_linear=[]
variance_linear=[]
error_linear=[]
accuracy_linear=[]
for c in np.round(np.arange(0.1,3.1,0.1),2):
    bias, variance, error ,accuracy= evaluate_model(C=c, solver='linear')
    bias_linear.append(bias)
    variance_linear.append(variance)
    error_linear.append(error)
    accuracy_linear.append(np.mean(accuracy))
    
bias_rbf=[]
variance_rbf=[]
error_rbf=[]
accuracy_rbf=[]
for c in np.round(np.arange(0.1,3.1,0.1),2):
    bias, variance, error ,accuracy= evaluate_model(C=c, solver='rbf')
    bias_rbf.append(bias)
    variance_rbf.append(variance)
    error_rbf.append(error)
    accuracy_rbf.append(np.mean(accuracy))
    
bias_sigmoid=[]
variance_sigmoid=[]
error_sigmoid=[]
accuracy_sigmoid=[]
for c in np.round(np.arange(0.1,3.1,0.1),2):
    bias, variance, error ,accuracy= evaluate_model(C=c, solver='sigmoid')
    bias_sigmoid.append(bias)
    variance_sigmoid.append(variance)
    error_sigmoid.append(error)
    accuracy_sigmoid.append(np.mean(accuracy))


# In[88]:


plt.figure(figsize=[20,12])

plt.subplot(2,2,1)
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),bias_linear,label="linear",marker='*')
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),bias_rbf,label="rbf")
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),bias_sigmoid,label="sigmoid")
plt.xlabel("C: Parametro de regularización")
plt.ylabel ("Bias")
plt.title("C vs bias")
plt.legend()
plt.grid(False)

plt.subplot(2,2,2)
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),variance_linear,label="linear",marker='*')
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),variance_rbf,label="rbf")
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),variance_sigmoid,label="sigmoid")
plt.xlabel("C: Parametro de regularización")
plt.ylabel ("Variance")
plt.title("Variance vs bias")
plt.legend()
plt.grid(False)


plt.subplot(2,2,3)
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),error_linear, label="linear", marker='*')
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),error_rbf,label="rbf")
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),error_sigmoid,label="sigmoid")
plt.xlabel("C: Parametro de regularización")
plt.ylabel ("Error")
plt.title("Error vs bias")
plt.legend()
plt.grid(False)


plt.subplot(2,2,4)
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),accuracy_linear, label="linear",marker='*')
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),accuracy_rbf,label="rbf")
plt.plot(np.round(np.arange(0.1,3.1,0.1),2),accuracy_sigmoid,label="sigmoid")
plt.xlabel("C: Parametro de regularización")
plt.ylabel ("Accuracy")
plt.title("Accuracy vs bias")
plt.grid(False)


plt.show()


# In[89]:


param_dist = {
    'C': np.logspace(-3, 3, 7),          # Rango para C
    'kernel': ['linear', 'rbf', 'poly'],  # Tipos de kernel
    'gamma': np.logspace(-3, 3, 7)       # Rango para gamma
}
svm = SVC()
random_search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print("Mejores hiperparámetros encontrados:")
print(best_params)


# In[90]:


model=SVC(kernel="linear",gamma=1000,C=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy de Support Vector Machine con los parametros de Random Search: {}%".format(accuracy*100))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión Support Vector Machine Multiclase')
plt.show()


# In[91]:


yTrain_pred=model.predict(X_train)
accuracy=accuracy_score(y_train, yTrain_pred)
print("Accuracy Training DataSet: {}%".format(accuracy*100))

cm = confusion_matrix(y_train, yTrain_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión Support Vector Machine Multiclase')
plt.show()


# In[92]:


model = SVC(kernel="linear",gamma=1000,C=1000)
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=train_sizes, cv=10, scoring='accuracy'
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.title("Learning Curve for Support Vector Machine - Best Model")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Accuracy")
plt.legend(loc="best")
plt.show()


# ## RANDOM FOREST

# In[93]:


def evaluate_model(estimators, depth):
    model = RandomForestClassifier(n_estimators=estimators, max_depth=depth,random_state=42)
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    bias = 1 - np.mean(accuracy_scores)
    variance = np.var(accuracy_scores, ddof=1)
    
    return bias, variance, 1 - np.mean(accuracy_scores),accuracy_scores

bias, variance, error ,accuracy= evaluate_model(estimators=1, depth=10)
print(f"Bias: {bias:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Error: {error:.2f}")
print("Accuracy: {}%".format(np.mean(accuracy)*100))


# In[94]:


model = RandomForestClassifier(n_estimators=1, max_depth=3,random_state=42)
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=train_sizes, cv=10, scoring='accuracy'
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.title("Learning Curve for Random Forest")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Accuracy")
plt.legend(loc="best")
plt.show()


# In[95]:


biasRFT=[]
varianceRFT=[]
errorRFT=[]
accuracyRFT=[]

for j in [1,5,10,15]:
    biasRF=[]
    varianceRF=[]
    errorRF=[]
    accuracyRF=[]
    for i in range(1,11):
        bias, variance, error ,accuracy= evaluate_model(estimators=j, depth=i)
        biasRF.append(bias)
        varianceRF.append(variance)
        errorRF.append(error)
        accuracyRF.append(np.mean(accuracy))
    biasRFT.append(biasRF)
    varianceRFT.append(varianceRF)
    errorRFT.append(errorRF)
    accuracyRFT.append(accuracyRF)
    
    
plt.figure(figsize=[20,15])
plt.subplot(2,2,1)
plt.plot(biasRFT[0], label="n_est=1")
plt.plot(biasRFT[1], label="n_est=5")
plt.plot(biasRFT[2], label="n_est=10")
plt.plot(biasRFT[3], label="n_est=15")
plt.ylabel("Bias")
plt.xlabel("Depth")
plt.title("Depth vs Bias")
plt.legend()

plt.subplot(2,2,2)
plt.plot(varianceRFT[0], label="n_est=1")
plt.plot(varianceRFT[1], label="n_est=5")
plt.plot(varianceRFT[2], label="n_est=10")
plt.plot(varianceRFT[3], label="n_est=15")
plt.ylabel("Variance")
plt.xlabel("Depth")
plt.title("Depth vs Variance")
plt.legend()

plt.subplot(2,2,3)
plt.plot(errorRFT[0],label="n_est=1")
plt.plot(errorRFT[1],label="n_est=5")
plt.plot(errorRFT[2],label="n_est=10")
plt.plot(errorRFT[3],label="n_est=15")
plt.ylabel("Error")
plt.xlabel("Depth")
plt.title("Depth vs Error")
plt.legend()

plt.subplot(2,2,4)
plt.plot(accuracyRFT[0], label="n_est=1")
plt.plot(accuracyRFT[1], label="n_est=5")
plt.plot(accuracyRFT[2], label="n_est=10")
plt.plot(accuracyRFT[3], label="n_est=15")
plt.ylabel("Accuracy")
plt.xlabel("Depth")
plt.title("Depth vs Accuracy")

plt.legend()
plt.show()


# In[96]:


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
param_dist = {
    'n_estimators': [1,5,10,15,20,50],  # Rango de valores para n_estimators
    'max_depth': [None] + list(np.arange(10, 110, 10)),  # Rango de valores para max_depth
    'min_samples_split': np.arange(2, 11),  # Rango de valores para min_samples_split
    'min_samples_leaf': np.arange(1, 11),  # Rango de valores para min_samples_leaf
    'max_features': ['auto', 'sqrt', 'log2', None],  # Opciones para max_features
    'bootstrap': [True, False],  # Opciones para bootstrap
    'random_state': [42]  # Semilla aleatoria fija para reproducibilidad
}
rf = RandomForestClassifier()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=100, scoring='accuracy',
    n_jobs=-1, cv=cv, random_state=42, verbose=2
)
random_search.fit(X_train, y_train)
print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)


# In[97]:


model=RandomForestClassifier(random_state=42,n_estimators=15,min_samples_split=6,min_samples_leaf=3,
                            max_features="sqrt",max_depth=70,bootstrap=True )
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy para Random Forest con Hiperparametros de Random Search:")
print(" {}%".format(accuracy*100))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión Random Forest Multiclase')
plt.show()


# In[98]:


yTrain_pred=model.predict(X_train)
accuracy=accuracy_score(y_train, yTrain_pred)
print("Accuracy Training DataSet: {}%".format(accuracy*100))

cm = confusion_matrix(y_train, yTrain_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión Random Forest Multiclase')
plt.show()


# In[99]:


model = RandomForestClassifier(random_state=42,n_estimators=15,min_samples_split=6,min_samples_leaf=3,
                            max_features="sqrt",max_depth=70,bootstrap=True )
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=train_sizes, cv=10, scoring='accuracy'
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.title("Learning Curve for Random Forest - Best Model")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Accuracy")
plt.legend(loc="best")
plt.show()


# # KNN 

# In[100]:


def evaluate_model(k):
    model = KNeighborsClassifier(n_neighbors=k)
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    bias = 1 - np.mean(accuracy_scores)
    variance = np.var(accuracy_scores, ddof=1)
    
    return bias, variance, 1 - np.mean(accuracy_scores),accuracy_scores

bias, variance, error ,accuracy= evaluate_model(k=2)
print(f"Bias: {bias:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Error: {error:.2f}")
print("Accuracy: {}%".format(np.mean(accuracy)*100))


# In[101]:


model = KNeighborsClassifier(n_neighbors=2)
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=train_sizes, cv=10, scoring='accuracy'
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.title("Learning Curve for KNN")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Accuracy")
plt.legend(loc="best")
plt.show()


# In[102]:


biasRF=[]
varianceRF=[]
errorRF=[]
accuracyRF=[]
for i in range(1,20):
    bias, variance, error ,accuracy= evaluate_model(k=i)
    biasRF.append(bias)
    varianceRF.append(variance)
    errorRF.append(error)
    accuracyRF.append(np.mean(accuracy))


# In[103]:


plt.figure(figsize=[20,12])

plt.subplot(2,2,1)
plt.plot(biasRF)
plt.ylabel("Bias")
plt.xlabel("Neighbors")
plt.title("Bias vs Neighbors")

plt.subplot(2,2,2)
plt.plot(varianceRF)
plt.ylabel("Variance")
plt.xlabel("Neighbors")
plt.title("Variance vs Neighbors")

plt.subplot(2,2,3)
plt.plot(errorRF)
plt.ylabel("Error")
plt.xlabel("Neighbors")
plt.title("Error vs Neighbors")

plt.subplot(2,2,4)
plt.plot(accuracyRF)
plt.ylabel("Accuracy")
plt.xlabel("Neighbors")
plt.title("Accuracy vs Neighbors")

plt.show()


# In[104]:


param_dist = {
    'n_neighbors': np.arange(1, 11),  # Rango de vecinos a considerar
    'weights': ['uniform', 'distance'],  # Peso de los vecinos ('uniform' o 'distance')
    'p': [1, 2]  # Parámetro de distancia (1 para Manhattan, 2 para Euclidiana)
}
knn = KNeighborsClassifier()
random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=10, cv=5)
random_search.fit(X, y)
print("Mejores hiperparámetros encontrados:", random_search.best_params_)


# In[105]:


model=KNeighborsClassifier(weights="distance",p=1,n_neighbors=6)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy para KNN con los Hiperparametros encontrados por Random Search: ")
print("{}%".format(accuracy*100))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión KNN')
plt.show()


# In[106]:


yTrain_pred=model.predict(X_train)
accuracy=accuracy_score(y_train, yTrain_pred)
print("Accuracy Training DataSet: {}%".format(accuracy*100))

cm = confusion_matrix(y_train, yTrain_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DrugY","drugA","drugB","drugC","drugX"],
            yticklabels=["DrugY","drugA","drugB","drugC","drugX"],cbar=False,
            annot_kws={"size": 16}, linewidths=0.5, linecolor="gray")
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión KNN')
plt.show()


# In[107]:


model = KNeighborsClassifier(weights="distance",p=1,n_neighbors=6)
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=train_sizes, cv=10, scoring='accuracy'
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.title("Learning Curve for KNN - Best Model")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Accuracy")
plt.legend(loc="best")
plt.show()

