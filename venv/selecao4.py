from sklearn import svm, datasets, metrics
from sklearn.feature_selection import RFECV, chi2, VarianceThreshold
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

db = datasets.load_digits()

X = db.data
Y = db.target
classes = db.target_names
k = 2

print(X.shape)
print('Remoção de Variância')
sel = VarianceThreshold()
X_new = sel.fit_transform(X,Y)

svc = svm.SVC(gamma='scale',kernel='linear')

print('Seleção dos melhores atributos')
atributos = [x for x in range(1,len(X_new[0])+1)]
sel = RFECV(estimator=svc,cv=5,scoring='accuracy')

X_new = sel.fit_transform(X_new,Y)

print("New Shape:",X_new.shape)

np.random.seed(0)

n_samples = len(X_new)
percentage = 0.75

order = np.random.permutation(n_samples)

X = X_new[order]
Y = Y[order]

print("Separação da Base")

Y_teste = Y[int(percentage*n_samples):]
X_teste = X[int(percentage*n_samples):]

Y_treino = Y[:int(percentage*n_samples)]
X_treino = X[:int(percentage*n_samples)]

scaler = StandardScaler()

print("Normalizando as bases.")
X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.fit_transform(X_teste)

parameters = {'kernel':('linear', 'rbf'), 'C':list(range(1,11))}

print("Busca dos parâmetros")

clf = GridSearchCV(svc,param_grid=parameters,cv=5)

clf.fit(X_treino,Y_treino)

print("Predição")
predicao = clf.predict(X_teste)

print("Resultados")
print(metrics.accuracy_score(Y_teste,predicao))
print(metrics.confusion_matrix(Y_teste,predicao))

