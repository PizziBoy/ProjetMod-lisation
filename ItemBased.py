# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:09:45 2020

@author: HP
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:28:46 2020

@author: HP
"""


from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import math
from sklearn.impute import KNNImputer


#------------------------------------------------------------------------------------------------------------------------

#                           Fonctions outils

#-----------------------------------------------------------------------------------------------------------------------



# Retourne la somme des élements d'une liste passée en paramètre
def sommeListe(liste):
    _somme = 0
    for i in liste:
        _somme = _somme + int(i)
    return _somme

# Retourne la moyenne des élements d'une liste passée en paramètre
def average(liste):
    assert len(liste) > 0
    return float(sommeListe(liste)) / len(liste)

# Retourne la similarité 
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

# Retourne la similarité
def cos_sim(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)



#-----------------------------------------------------------------------------------------------------------------------

#                               Fonctions pour la prédiction

#----------------------------------------------------------------------------------------------------------------------- 
 
 
      
# Retourne la liste des 10 meilleures similarités par rapport a un item donné
def simList(dataSet, item):
    simListByUser = {}
    #simListUser = {}
    for i in range(0, 1000):
        liste = etLogique(item, dataSet[i])
        if (cos_sim(liste[0], liste[1]) >= 1):
            simListByUser[i] = -1
        else:
            simListByUser[i] = cos_sim(liste[0], liste[1])
    return dict(reversed(sorted(simListByUser.items(), key=lambda item: item[1])))


# Cette fonction retourne une liste comprenant les deux listes passées en paramètre avec les valeurs -1 supprimées
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

def removeDonnéeManquante(item):
    list = []
    for i in range(item.size):
        if not(i == -1):
            list.append(i)
    return list


#Renvoie les indices des données manquantes d'un item passé en paramètre
def rechercheDonnéeManquante(item):
    list = []
    s = 0
    for i in item:
        if (i == -1):
            list.append(s)
        s += 1
    return list

#Renvoie une liste de toutes les notes item d'un même utilisateur
def noteUserList(simListItem, user):
    list = []
    for i in simList.keys():
        list.append(user[i])
    return list

#Calcule une note en fonction des similarités 
def calculNote(simList, noteUserList):
        resultPoids = 0
        sommeSim = 0
        simList = etLogique(simList, noteUserList)[0]
        noteUserList = etLogique(simList, noteUserList)[1]
        i = 0
        for i in range(0, 4):
            resultPoids += simList[i] * noteUserList[i]
            sommeSim += simList[i]
        return resultPoids/sommeSim



#-----------------------------------------------------------------------------------------------------------------
        
#                                   Prediction de notes
        
#-----------------------------------------------------------------------------------------------------------------

#J'ouvre les dataSet toy_incomplet et toy_complet
dataSetJouet = pd.read_csv("toy_incomplet.csv", sep=' ', header = None)
dataSetJouetComplet = pd.read_csv("toy_incomplet.csv", sep=' ', header = None)

#Prédis les valeurs manquantes d'un item
def predictItem(item, dataSetJouet):
    listeNotesManquantes = rechercheDonnéeManquante(item)
    listeSimilarite = simList(dataSetJouet, item)
    for i in listeNotesManquantes:
        listeNotesCourantes = noteUserList(listeSimilarite, dataSetJouet.loc[i])
        item[i] = calculNote(listeSimilarite, listeNotesCourantes, item[i])
    return item
        
# Prédis les valeurs manquantes pour tout le dataSet
def predictAll(dataSetJouet):
    newDf = pd.DataFrame(lines = range(100))
    for i in range(0, 999):
        tmp = pd.Series(data = predictItem(dataSetJouet.loc[i], dataSetJouet))
        newDf = pd.concat([newDf, tmp.to_frame().T], ignore_index=True)
    return newDf

# Retourne la différence entre nos predictions et les veritables notes 
def diff(dataSet1, dataSet2):
    newDf = pd.DataFrame()
    for i in range(100):
        newDf[i] = np.where(dataSet1[i] == dataSet2[i], 0, dataSet1[i] - dataSet2[i])
    return newDf


# Je calcule la prédiction de notes pour l'Item 1
print(predictItem(dataSetJouet[0], dataSetJouet))

# Je calcule l'écart de mes predictions avec la valeurs réelles
#print(diff(dataSetJouetComplet,predictItem(dataSetJouet[0], dataSetJouet)))


# Methode des KNN
#imputer = KNNImputer(n_neighbors=5, missing_values=-1)
#print(imputer.fit_transform(dataSetJouet))
#print(diff(dataSetJouetComplet,predictItem(dataSetJouet[0], dataSetJouet)))
