from sklearn import svm, datasets, metrics
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
import numpy as np
import matplotlib.pyplot as plt


db = datasets.load_digits()

X = db.data
Y = db.target

sel = VarianceThreshold(threshold=(0.9*(1-0.9)))

X_new = sel.fit_transform(X)

print(len(X[0]))
print(len(X_new[0]))

variancia = sel.variances_
features = list(range(1,len(X[0])+1))

plt.title("Variância sobre os atributos de Digitos.")
plt.xlabel("Atributo")
plt.ylabel("Variância")

plt.bar(features,variancia)

for index, data in enumerate(features):
    plt.text(x=index+0.6,y=variancia[index]+0.5,s='{0:.1f}'.format(variancia[index]),fontdict=dict(fontsize=8),rotation=90)

plt.show()