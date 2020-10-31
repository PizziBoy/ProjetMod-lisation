
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
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

#Retourne une liste des 10 meilleures similarités rapport a un utilisateur donné
def simList(dataSet, utilisateur, nUtilisateurs):
    simListByUser = {}
    simListUser = {}
    for i in range(0, 100):
        liste = etLogique(utilisateur, dataSet.loc[i])
        if (cos_sim(liste[0], liste[1]) >= 1):
            simListByUser[i] = -1
        else:
            simListByUser[i] = cos_sim(liste[0], liste[1])
    return dict(reversed(sorted(simListByUser.items(), key=lambda item: item[1])))



def etLogique(u0, u1):
    liste0 = []
    liste1 = []
    les2Listes = []
    for i in range(0, len(u0)):
        if (u0[i] != -1 and u1[i] != -1):
            liste0.append(u0[i])
            liste1.append(u1[i])
    les2Listes.append(liste0)
    les2Listes.append(liste1)
    return les2Listes




#Renvoie les indices des données manquantes d'un utilisateur
def rechercheDonnéeManquante(utilisateur):
    list = []
    s = 0
    for i in utilisateur:
        if (i == -1):
            list.append(s)
        s += 1
    return list

#Renvoie une liste de toutes les notes utilisateurs d'un même item
def noteList(simList, item):
    list = []
    for i in simList.keys():
        list.append(item[i])
    return list

#Calcule une note en fonction de les similarités et les notes (10 pour 10 premiers utilisateurs)
def calculNote(simList, noteList):
        resultPoids = 0
        sommeSim = 0
        simList = etLogique(simList, noteList)[0]
        noteList = etLogique(simList, noteList)[1]
        i = 0
        for i in range(0, 4):
            resultPoids += simList[i] * noteList[i]
            sommeSim += simList[i]
        return resultPoids/sommeSim


#Prédis les valeurs manquantes d'un utilisateur
def predictUtilisateur(utilisateur, dataSetJouet):
    listeNotesManquantes = rechercheDonnéeManquante(utilisateur)
    listeSimilarite = simList(dataSetJouet, utilisateur, 10)
    for i in listeNotesManquantes:
        listeNotesCourantes = noteList(listeSimilarite, dataSetJouet[i])
        utilisateur[i] = calculNote(listeSimilarite, listeNotesCourantes)
    return utilisateur

def predictAll(dataSetJouet):
    newDf = pd.DataFrame(columns = range(1000))
    for i in range(0, 100):
        tmp = pd.Series(data = predict(dataSetJouet.loc[i], dataSetJouet))
        newDf = pd.concat([newDf, tmp.to_frame().T], ignore_index=True)
    return newDf

def diff(dataSet1, dataSet2):
    newDf = pd.DataFrame()
    for i in range(1000):
        newDf[i] = np.where(
        dataSet1[i] == dataSet2[i], 0, dataSet1[i] - dataSet2[i])
    return newDf

#J'ouvre le dataSet toy_incomplet
dataSetJouet = pd.read_csv("toy_incomplet.csv", sep=' ', header = None)
dataSetJouetComplet = pd.read_csv("toy_complet.csv", sep=' ', header = None)
#.loc[0] = ligne 0 et [0] = colonne 0
#print(calculNote([0.72, 0.55, 0.33], [2, 3, 5]))
#print(simList(dataSetJouet, dataSetJouet.loc[0], 10))
print(predictUtilisateur(dataSetJouet.loc[0], dataSetJouet))

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, missing_values=-1)

print(diff(dataSetJouetComplet,predictUtilisateur(dataSetJouet.loc[0], dataSetJouet) ))
