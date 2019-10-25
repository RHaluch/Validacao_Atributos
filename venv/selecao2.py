from sklearn import svm, datasets, metrics
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
import numpy as np
import matplotlib.pyplot as plt


db = datasets.load_digits()

X = db.data
Y = db.target
k = 20

print(X.shape)
print('Remoção de Variância')
sel = VarianceThreshold()
X_new = sel.fit_transform(X,Y)

print(X_new.shape)

print('Seleção dos melhores atributos')
atributos = [x for x in range(1,len(X_new[0])+1)]
sel = SelectKBest(chi2,k=k)

X_new = sel.fit_transform(X_new,Y)


print(X_new.shape)

plt.title("Seleção dos K = {0} melhores atributos.".format(k))
plt.xlabel("Atributo")
plt.ylabel("Score - Normalizado")

scores = sel.scores_
print(scores)
print(np.linalg.norm(scores))
normalized = scores / np.linalg.norm(scores,ord=np.inf)
print(normalized)
plt.bar(atributos,normalized)

for index, data in enumerate(scores):
    plt.text(x=index+0.6,y=normalized[index]+0.01,s='{0:.2f}'.format(normalized[index]),fontdict=dict(fontsize=8),rotation=90)

plt.show()
