import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import KNNImputer
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np
import math



def sommeListe(liste):
    _somme = 0
    for i in liste:
        _somme = _somme + int(i)
    return _somme


def average(x):
    assert len(x) > 0
    return float(sommeListe(x)) / len(x)

def pearson(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = int(x[idx]) - avg_x
        ydiff = int(y[idx]) - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)


def cos_sim(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

#J'ouvre le dataSet toy_complet
f = open("toy_complet.csv")
fPrime = pd.read_csv("toy_complet.csv", sep=' ', header = None)

print(df[0])

toy_complet = csv.reader(f)
dataSet = []

#Je mets dans la liste dataSet chaque ligne que contient toy_complet
for line in toy_complet:
  dataSet.append(line)

for column in toy_complet:
    print(column)



#Je sélectionne les utilisateurs qui correspondent aux lignes
utilisateur0 = dataSet[0]
utilisateur1 = dataSet[1]

#Je split les éléments enregistrés dans la premiere case de ma liste Utilisateurn
utilisateur0List = utilisateur0[0].split()
utilisateur1List = utilisateur1[0].split()


#Je transforme en la liste en matrice
utilisateur0 = np.array(utilisateur0List)
utilisateur1 = np.array(utilisateur1List)

utilisateur0Float=np.array(utilisateur0,dtype=float)
utilisateur1Float=np.array(utilisateur1,dtype=float)

utilisateur0 = utilisateur0.reshape(1, -1)
utilisateur1 = utilisateur1.reshape(1, -1)

#imputer = KNNImputer(missing_values = -1, n_neighbors = 1)
#print(imputer.fit_transform(utilisateur0))


#print(cosine_similarity(utilisateur0, utilisateur1))
#print(cos_sim(utilisateur0Float, utilisateur1Float))
#print(pearson(utilisateur0List, utilisateur1List))
#print(pearsonr(utilisateur0Float, utilisateur1Float))
