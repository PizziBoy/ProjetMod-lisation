
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import math
from sklearn.impute import KNNImputer
import plotly.express as px
import matplotlib.pyplot as plt

#------------------------------------------------------------------------
#                             Fonctions outils !!
#------------------------------------------------------------------------
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




#------------------------------------------------------------------------
#                       Fonctions pour la prédiction (test)
#------------------------------------------------------------------------


#J'ouvre les dataSet
dataSetJouet = pd.read_csv("toy_incomplet.csv", sep=' ', header = None)
dataSetJouetComplet = pd.read_csv("toy_complet.csv", sep=' ', header = None)
#.loc[0] = ligne 0 et [0] = colonne 0



#Retourne un dictionnaire trié dans l'ordre des meilleures similarités pour un utilisateur donné et le reste du dataSet
def simList(dataSet, utilisateur):
    simListByUser = {}
    list = []
    for i in range(0, 100):
        liste = etLogique(utilisateur, dataSet.loc[i])
        if (cos_sim(liste[0], liste[1]) >= 1):
            simListByUser[i] = -1
        else:
            simListByUser[i] = cos_sim(liste[0], liste[1])
    dic = dict(reversed(sorted(simListByUser.items(), key=lambda item: item[1])))
    list = [(k, v) for k, v in dic.items()]
    return list



#Execute un &logique entre les notes de 2 utilisateur afin de retourner uniquement les notes existantes des 2 côtés
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



#Enleve tous les -1 dans les notes d'un utilisateur (une liste)
def removeDonnéeManquante(utilisateur):
    list = []
    for i in utilisateur:
        if (i != -1):
            list.append(i)
    return list



#Renvoie les indices d'où se situent les données manquantes d'un utilisateur
def rechercheDonnéeManquante(utilisateur):
    list = []
    s = 0
    for i in utilisateur:
        if (i == -1):
            list.append(s)
        s += 1
    return list



#Calcule une note en fonction des similarités et des notes utilisateur (moyenne pondérée de 0 au n ieme meilleur)
def calculNote(simList, notesItem):
        n = 4
        resultPoids = 0
        sommeSim = 0
        i = 0
        for i in range(0, 15):
            if (notesItem[simList[i][0]] != -1 and simList[i][1] != -1):
                resultPoids += simList[i][1] * notesItem[simList[i][0]]
                sommeSim += simList[i][1]
        return resultPoids/sommeSim



#Calcule une note même méthode mais avec prise en compte de la sévérité utilisateur
def calculNoteAvecSeverite(simList, notesItem, utilisateurConcerné):
        resultPoids = 0
        sommeSim = 1
        i = 0
        avgU = average(removeDonnéeManquante(utilisateurConcerné));
        for i in range(15):
            if (notesItem[simList[i][0]] != -1 and simList[i][1] != -1):
                resultPoids += simList[i][1] * (notesItem[simList[i][0]] - avgU)
                sommeSim += simList[i][1]
        return avgU + resultPoids/sommeSim



#Prédis les valeurs manquantes d'un utilisateur
def predictUtilisateur(utilisateur, dataSetJouet, severite):
    listeNotesManquantes = rechercheDonnéeManquante(utilisateur)
    listeSimilarite = simList(dataSetJouet, utilisateur)
    for i in listeNotesManquantes:
        listeNotesCourantes = dataSetJouet[i]
        if (severite):
            utilisateur[i] = calculNoteAvecSeverite(listeSimilarite, listeNotesCourantes, utilisateur)
        else:
            utilisateur[i] = calculNote(listeSimilarite, listeNotesCourantes)
    return utilisateur



#Prédis tous le dataSet entier
def predictAll(dataSetJouet, severite):
    newDf = pd.DataFrame(columns = range(1000))
    for i in range(0, 100):
        tmp = pd.Series(data = predictUtilisateur(dataSetJouet.loc[i], dataSetJouet, severite))
        newDf = pd.concat([newDf, tmp.to_frame().T], ignore_index=True)
    return newDf



#Retourne un dataSet avec la différence calculée entre 2 mêmes dataSet mais pas les mêmes techniques
def diff(dataSet1, dataSet2):
    newDf = pd.DataFrame()
    for i in range(1000):
        newDf[i] = np.where(
        dataSet1[i] == dataSet2[i], 0, abs(dataSet1[i] - dataSet2[i]))
    return newDf



#print(calculNote([0.72 , 0.33], [2, 5]))
#print(simList(dataSetJouet, dataSetJouet.loc[0], 10))

print(predictAll(dataSetJouet, False))

filename = "predict_toy_4meilleurs.csv"
dict = simList(dataSetJouet, dataSetJouet.loc[0])
#print(type(dict[0][0]))


#imputer = KNNImputer(n_neighbors=5, missing_values=-1)
#print(imputer.fit_transform(dataSetJouet))
#plt.plot( range(1000*100), diff(dataSetJouetComplet, predictAll(dataSetJouet)).values.tolist())


#print(diff(dataSetJouetComplet,predictUtilisateur(dataSetJouet.loc[0], dataSetJouet) ))
