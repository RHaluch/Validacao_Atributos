from sklearn import svm, datasets, metrics
from sklearn.model_selection import validation_curve, learning_curve
import numpy as np
import matplotlib.pyplot as plt


db = datasets.load_digits()

X = db.data
Y = db.target

np.random.seed(0)

n_samples = len(X)
percentage = 0.75

order = np.random.permutation(n_samples)

X = X[order]
Y = Y[order]

Y_teste = Y[int(percentage*n_samples):]
X_teste = X[int(percentage*n_samples):]

Y_treino = Y[:int(percentage*n_samples)]
X_treino = X[:int(percentage*n_samples)]

C = [x for x in range(1,11)]

clf = svm.SVC(gamma='scale')

train_scores, validation_scores = validation_curve(clf,X_treino,Y_treino,"C",C,verbose=3,cv=5)
train_sizes, train_scores_lc, validation_scores_lc = learning_curve(clf,X_treino,Y_treino,cv=5)

train_scores_mean = np.mean(train_scores,axis=1)
train_scores_std = np.std(train_scores,axis=1)
validation_scores_mean = np.mean(validation_scores,axis=1)
validation_scores_std = np.std(validation_scores,axis = 1)

train_scores_mean_lc = np.mean(train_scores_lc,axis=1)
train_scores_std_lc = np.std(train_scores_lc,axis=1)
validation_scores_mean_lc = np.mean(validation_scores_lc,axis=1)
validation_scores_std_lc = np.std(validation_scores_lc,axis = 1)

plt.subplot(1,2,1)
plt.title("Curva de validação com SVM")
plt.xlabel("C")
plt.ylabel("Score")

min_all = np.min([np.min(train_scores),np.min(validation_scores)])

plt.ylim(min_all,1.001)
plt.grid()
param_range = list(range(1,len(C)+1))
#Largura da linha
lw = 2
plt.plot(param_range,train_scores_mean,'o-',label='Training Score',lw=lw)
plt.fill_between(param_range,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.2,lw=lw)

plt.plot(param_range,validation_scores_mean,'o-',label='Validation Score',lw=lw)
plt.fill_between(param_range,validation_scores_mean-validation_scores_std,validation_scores_mean+validation_scores_std,alpha=0.2,lw=lw)

plt.legend(loc="best")

plt.subplot(1,2,2)

plt.title("Curva de aprendizado com SVM")
plt.xlabel("Exemplos de Treino")
plt.ylabel("Score")
plt.grid()

min_all = np.min([np.min(train_scores_lc),np.min(validation_scores_lc)])

plt.ylim(min_all,1.001)

#Largura da linha
lw = 2
plt.plot(train_sizes,train_scores_mean_lc,'o-',label='Training Score',lw=lw)
plt.fill_between(train_sizes,train_scores_mean_lc-train_scores_std_lc,train_scores_mean_lc+train_scores_std_lc,alpha=0.2,lw=lw)

plt.plot(train_sizes,validation_scores_mean_lc,'o-',label='Validation Score',lw=lw)
plt.fill_between(train_sizes,validation_scores_mean_lc-validation_scores_std_lc,validation_scores_mean_lc+validation_scores_std_lc,alpha=0.2,lw=lw)

plt.legend(loc="best")

plt.show()