# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:00:00 2020

@author: Alexandre NEROT
contact : alexandre@nerot.net

Ceci correspond aux fonctions utilisées pour traiter les données dans le cadre du projet ODIASP
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import scipy
from scipy.ndimage import zoom, center_of_mass
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as MAGIC
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
import os
import pandas
from PIL import Image
import random
from shutil import copy2, move
from copy import copy
import skimage.io as io
import skimage.transform as trans
from alive_progress import alive_bar
import datetime
import scipy
import openpyxl
import time



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K


from Settings import COMPUTE_CAPACITY


def DirVerification (name,DossierProjet=None,verbose = 1):
    if os.path.exists(name):
        if verbose ==1 :
            print("Le dossier " + str(name) + " existe déja : ", os.path.abspath(name))
        return name
    
    else :
        if DossierProjet!=None:
            dir_path = os.path.join(DossierProjet,name)
        else : dir_path = name

        if os.path.exists(dir_path):
            if verbose ==1 :
                print("Le dossier " + str(name) + " existe déja : ")
                print(name, " : ", dir_path)
        else : 
            os.mkdir(dir_path)
            if verbose == 1 :
                print("Création du dossier " + str(name)+ " : ")
                print(name, " : ", dir_path)
        return dir_path


        
#___________________________________________________________________________________________
#___________________FONCTIONS POUR LA CREATION DES DATASETS ________________________________
#___________________________________________________________________________________________

       
def readCSV(csv_path,name=None,indexing=None):
    """
    Fonction simple pour lire le CSV et le garder en mémoire sous la forme d'un datafile, plus facilement lisible en utilisant pandas
    si on rentre name (un des fichiers numpy disponibles), la fonction affiche cette valeur
    On peut rentrer un string pour l'arg indexing pour demander a classer selon la colonne.
    """
    df=pandas.read_csv(csv_path, delimiter=",",dtype=str)
    if indexing != None :
        df.set_index(indexing, inplace=True)
    if name:
        print(df.loc[name])
    return df

    
#___________________________________________________________________________________________
#___________________FONCTIONS POUR IMPORTER LES FICHIERS DICOM______________________________
#___________________________________________________________________________________________



def fast_scandir(dir):
    """
    Prend un dossier contenant atant de sous-dossiers et sous-sous-dossiers que voulu et en crée la liste des sous dossiers.
    Utile pour généraliser une fonction adaptée à un dossier à autant de dossiers que voulus en une seule fois.
    Rq : le dossier dir n'est pas inclus dans la liste
    
    Parameters
    ----------
        - dir : string, chemin vers le dossier racine
        
    Returns
    -------
        - subfloders : liste, contient tous les sous-dossiers
    
    """
    subfolders= [f.path for f in os.scandir(dir) if f.is_dir()]
    for dir in list(subfolders):
        subfolders.extend(fast_scandir(dir))
    return subfolders


def TESTINGPRED(prediction,classes,mode="ProportionGlobale", nombredecoupes=1, numerocoupeinitial = 0, verbose=1):
    longueur = len(prediction)
    proportion =0
    if numerocoupeinitial>longueur: #utilisé à des fins de DEBUGGING uniquement : a priori non possible.
        print("error") 
        numerocoupeinitial=longueur
    if (nombredecoupes+numerocoupeinitial)>longueur:
        nombredecoupes = longueur - numerocoupeinitial
    
    if mode=="ProportionGlobale":
        #test de l'ensemble du volume
        nombredecoupes =longueur
        for i in range(0,longueur):
            proportion += prediction[i]
        
        if verbose >0 :   
            for item in range(0,len(classes)):
                print(mode,"sur",nombredecoupes,"coupes de",classes[item], '{:.3f}'.format(proportion[item]/nombredecoupes))

    if mode=="ProportionFinale":
        #test des "nombredecoupes" dernières coupes
        if nombredecoupes>longueur : #au cas où l'on ait choisi trop de coupes
            nombredecoupes=longueur
        for i in range(numerocoupeinitial,nombredecoupes+numerocoupeinitial):
            proportion += prediction[-i]
        
        if verbose >0 :   
            for item in range(0,len(classes)):
                print(mode,"sur",nombredecoupes,"coupes de",classes[item], '{:.3f}'.format(proportion[item]/nombredecoupes))
    
    if mode=="ProportionInitiale":
        #test des "nombredecoupes" dernières coupes
        if nombredecoupes>longueur : #au cas où l'on ait choisi trop de coupes
            nombredecoupes=longueur
        for i in range(numerocoupeinitial,nombredecoupes+numerocoupeinitial):
            proportion += prediction[i]
        
        if verbose >0 :   
            for item in range(0,len(classes)):
                print(mode,"sur",nombredecoupes,"coupes de",classes[item], '{:.3f}'.format(proportion[item]/nombredecoupes), "à partir de la coupe", numerocoupeinitial)
    
    return proportion/nombredecoupes


    
    
    
def import_dicom_to_abdopelv(rootdir,
                             metadata, 
                             csv_path, 
                             save=False, 
                             model = None,
                             verbose = 2,
                             Compute_capacity = COMPUTE_CAPACITY,
                             CUPY = True
                             ):
    """
    Cette fonction charge un dossier contenant des fichiers DICOM. Elle peut s'arréter automatiquement si les métadonnées ou le nombre d'image ne correspondent pas.
    Sinon elle charge toutes les coupes pour les analyser et les labeliser. Si le scanner contient une partie abdomonopelvienne il est mis en mémoire (voire sauvegarder si "save" est rempli) puis les infos sont enregistrées dans le csv

    Parameters
    ----------
        - rootdir : string, chemin complet vers un dossier contenant des fichiers DICOM
        - metadata : correspond à la liste des Metadata que l'on veut récupérer dans les DICOM 
        - csv_path : chemin (complet) vers le csv pour enregistrer les metadatas du scanner. Le CSV sera créé automatiquement si celui-ci n'existe pas encore
        - save=False, False ou string, indique le chemin pour sauvegarder les volumes .npy correspondant aux scanners chargés. Optionnel (False : garde les numpy en mémoire.) Remarque : les numpy sont lourds, plus que les DICOM (qui sont compressés), environ 700Mo pour un scan abdo, on peut vite remplir un disque dur en utilisant cete fonction sur un dossier comprenant beaucoup de scanners !
        - model : le model de labelisation
        - Compute_capacity = COMPUTE_CAPACITY : int, la capacité de calcul des gpu, voir site nvidia. Pour adapter automatiquement le calcul réalisé par le reseau en fonction de la capacité de l'ordinateur
        - verbose : qt de verbose. 0 pas de verbose ; 1 : texte seulement ; 2 : texte et images (met le terminal en pause à chaque image)
        
    Returns
    -------
    Si le scanner a été chargé :
        - volume_numpy : le volume chargé.
        - perduSUP, perduBAS : les coupes non chargées sur la partie supérieure et la partie basse du scanner
        - facteur : facteur d'agrandissement
        - NOMFICHIER : le nom du fichier qui a été utilisé pour remplir le .csv
    
    Si la fonction s'est arrêtée spontanément :
        - volume_numpy = None
        - perduSUP = "Arret"
        - perduBAS = 0
        - facteur = "niveau1" ou "niveau2"
        - None
        
    Notes
    -----

    
    """
    classtotest = ['Abdomen','Cervical','Diaphragme','Membreinf','Pelvien','Thorax']
    
    #Verification que le dossier de sortie existe
    if save != False :
        save  = DirVerification (save, verbose = 0)
    
    #Ecriture du csv
    if os.path.isfile(csv_path)==False:
        titres=""
        titres += "Name"
        for ele in metadata:
            titres += ","+ str(ele)
        titres += ",OriginalSlices,DeletedSlices,Facteur,Path,L3Position,Certitude,L3Original"
        if verbose>0:print("Creation du csv")
        with open(csv_path,"wt", encoding="utf-8") as file :
            file.write(titres)
            file.close()
 
    #Création des variables dont nous allons avoir besoin :
    stopNow = False
    perduSUP = 0
    perduBAS = 0
    facteur = 1
    erreur = " "
    volume_numpy = np.empty(0)
    inter = {}
    start = time.perf_counter()
    
    
    list_files = os.listdir(rootdir)
    if len(list_files) <150:
        erreur = " Pas assez de coupes ({} fichiers)".format(len(list_files))
        perduSUP = "Arret"
        facteur = "niveau1"
        if verbose>0:print("   Arrêt précoce de niveau 1. Les images n'ont pas été chargées : ",erreur)
        return volume_numpy, perduSUP, perduBAS, facteur, None #le volume numpy est donc vide si le dossier n'avait pas les informations requises.

    else :
        test1 = 1
        test2 = 100
        
        echantillon1 = os.path.join(rootdir, list_files[test1])
        echantillon2 = os.path.join(rootdir, list_files[test2])
        
        while os.path.isdir(echantillon1):
            test1 +=1
            echantillon1 = os.path.join(rootdir, list_files[test1])
        
        while os.path.isdir(echantillon1):
            test2 +=1
            echantillon2 = os.path.join(rootdir, list_files[test2])
                     
        if not os.path.isdir(echantillon1):
            _ds_1 = pydicom.dcmread(echantillon1,force =True)
            
            
            """
            Donnons un nom à cette série
            """
            
            if (0x20, 0x000E) in _ds_1:
                NameSerie = str(_ds_1["SeriesInstanceUID"].value)
                NameSerie = NameSerie.replace('.','')
                NameSerie = NameSerie[-30:]
            else :
                NameSerie = "000000000000000000000000000000"
            
            NOMFICHIER = str(os.path.basename(rootdir))+"_"+NameSerie+r".npy"
            
            """
            Verifions que cette serie n'a pas deja ete analysee en quel cas on s'arrête.
            """
            if os.path.isfile(csv_path)==True:
                df = readCSV(csv_path,name=None,indexing="Name")
                if df.index.str.contains(NameSerie).any() :
                    erreur = " Scanner déjà analysé."                  
                    if verbose>0:print("   Arrêt précoce de niveau 1. Les images n'ont pas été chargées : ",erreur)
                    perduSUP = "Arret"
                    facteur = "niveau1"
                    return volume_numpy, perduSUP, perduBAS, facteur, None
                    
            """
            Nous essayons de déterminer s'il s'agit d'un scanner AP ou TAP à partir uniquement des métadonnées du dicom
            Cela permet si ce n'est pas le cas de réaliser une fonction rapide, qui ne charge quasiment aucune image.
            """        
            if (0x08, 0x60) in _ds_1:
                modalite = _ds_1["Modality"].value
                if str(modalite) != "CT": #Limitation si ce n'est pas un scanner !
                    erreur += " Le fichier DICOM n'est pas un scanner."
                    stopNow = True
            
            if (0x18, 0x50) in _ds_1:
                thickness = _ds_1["SliceThickness"].value
                if thickness >2.5: #Limitation si coupe trop épaisses : MIP...etc
                    erreur += " Epaisseur de coupe trop importante."
                    stopNow = True
                    
            if (0x28, 0x1050) in _ds_1:
                WindowCenter = _ds_1["WindowCenter"].value
                try :
                    if WindowCenter <0:
                        erreur += " Scanner Pulmonaire." #Limitation si fenetre pulmonaire
                        stopNow = True
                except :
                    erreur += " Erreur inconnue sur le WindowCenter :" + str(WindowCenter) + "."
                    stopNow = True 
                if WindowCenter ==500:
                    erreur += " Fenetrage Os." #Limitation si fenetre os (car trop de grain)
                    stopNow = True
                    
            if (0x18, 0x15) in _ds_1:
                BodyPartExamined = _ds_1["BodyPartExamined"].value
                if "HEAD" in str(BodyPartExamined) : #Limitation si imagerie cerebrale
                    erreur += " BodyPartExamined : "+str(BodyPartExamined)
                    stopNow = True
            
            #Verification de l'âge
            if (0x10, 0x10) in _ds_1:
                Age = _ds_1["PatientAge"].value
                if Age[-1:] != "Y" :
                    erreur += " Patient mineur : "+str(Age)
                    stopNow = True
                else : 
                    try : 
                        Age = Age[:-1]
                        if float(Age) < 18. : #Limitation si patient mineur
                            erreur += " Patient mineur : "+str(Age)
                            stopNow = True
                    except TypeError:
                        print("Erreur dans la lecture de l'âge chez ce patient :", Age, type(Age))
                        pass
                
            if (0x18, 0x1160) in _ds_1:
                BodyPartExamined = _ds_1["FilterType"].value
                if "HEAD" in str(BodyPartExamined) : #Limitation si imagerie cerebrale (autre moyen de verification)
                    erreur += " FilterType : " + str(BodyPartExamined)
                    stopNow = True
                    
            if (0x8, 0x103E) in _ds_1:
                BodyPartExamined = _ds_1["SeriesDescription"].value
                if "Crane" in str(BodyPartExamined) : #Limitation si imagerie cerebrale (autre moyen de verification)
                    erreur += " Scanner Cranien."
                    stopNow = True
                    
        if not os.path.isdir(echantillon2):
            _ds_2 = pydicom.dcmread(echantillon2,force =True,specific_tags =["ImagePositionPatient","SliceThickness"])
        position1 = [5.,10.,15.] 
        position2 = [5.,10.,15.] #Lecture de la position pour ne pas prendre de MPR coro ou sag
        if (0x20, 0x32) in _ds_1:
            position1 = _ds_1["ImagePositionPatient"].value
        if (0x20, 0x32) in _ds_2:
            position2 = _ds_2["ImagePositionPatient"].value
        if position1[0] != position2[0]:
            erreur += " Reconstruction Sagittale."
            stopNow = True
        if position1[1] != position2[1]:
            erreur += " Reconstruction Coronale."
            stopNow = True
        
    if stopNow == True:
        """
        Si le scanner n'est ni AP ni TAP la fonction s'arrête dès maintenant.
        """
        
        if verbose>0:print("   Arrêt précoce de niveau 1. Les images n'ont pas été chargées : ",erreur)
        perduSUP = "Arret"
        facteur = "niveau1"
        
        
        if verbose>1:print("Mise à jour du fichier csv :", csv_path)
        #Essai remplir csv meme pour les arrets
        values=[]
        for de2 in metadata:
            if de2 in _ds_1:
                if _ds_1[de2].VR == "SQ":
                    values = values + "sequence"
                elif _ds_1[de2].name != "Pixel Data":
                    _ds = str(_ds_1[de2].value)[:64]
                    raw_ds = _ds.replace('\n','__')
                    raw_ds = raw_ds.replace('\r','__')
                    raw_ds = raw_ds.replace('\t',"__")
                    raw_ds = raw_ds.replace(',',"__")
                    values.append(raw_ds)
            
            
        end = time.perf_counter()
        Timing = end-start
        
        dictMETADATAS = dict(zip(metadata, values))
        
        dictODIASP = {'Name' : NOMFICHIER, "Duree" : Timing, "Erreur" : erreur,'OriginalSlices' : len(list_files), 'Path' : rootdir, "Archive" : None}
        dict3 = {**dictMETADATAS , **dictODIASP}
        df=pandas.read_csv(csv_path, delimiter=",")
        modDfObj = df.append(dict3, ignore_index=True)
        modDfObj.to_csv(csv_path, index=False)
        
        return volume_numpy, perduSUP, perduBAS, facteur, None #le volume numpy est donc vide si le dossier n'avait pas les informations requises.
        
        
        
        
        
    if stopNow == False:
        """
        Maintenant que l'on a arrêté la fonction précocement selon certains criteres, regardons la liste des images
        """
        for f in list_files:
            if not os.path.isdir(f):
                f_long = os.path.join(rootdir, f)
                _ds_ = pydicom.dcmread(f_long,specific_tags =["ImagePositionPatient","SliceThickness"])
                inter[f_long]=_ds_.ImagePositionPatient[2]
        inter_sorted=sorted(inter.items(), key=lambda x: x[1], reverse=True) #il faut les trier dans l'ordre de lasequece d escanner (ce qui n'est pas l'ordre alphabetique du nom des fichiers)
        liste_fichiers=[x[0] for x in inter_sorted]

        path_img1=liste_fichiers[0]
        ds_img1=pydicom.dcmread(path_img1,stop_before_pixels=True)
        x_dim=int(ds_img1[0x28,0x10].value)
        y_dim=int(ds_img1[0x28,0x11].value)
        

        nbcoupes = len(liste_fichiers)
        if verbose>0:print(len(liste_fichiers), " fichiers trouvés pour ce scanner")
        
        if verbose>0:print("Creation d'un volume echantillon pour labelisation")
        x_dimDIV=x_dim/4
        y_dimDIV=y_dim/4
        ratioECHANTILLONAGE = 5 #Nous allons tester le volume à cet intervalle de coupe
        hauteur = len(liste_fichiers)//ratioECHANTILLONAGE 
        volume_pour_label=np.zeros((hauteur,int(x_dimDIV),int(y_dimDIV),3))
        for k in range (0,hauteur):
            
            dicom_file = pydicom.read_file(liste_fichiers[ratioECHANTILLONAGE*k])
            img_orig_dcm = (dicom_file.pixel_array)

            slope=float(dicom_file[0x28,0x1053].value)
            intercept=float(dicom_file[0x28,0x1052].value)
            img_modif_dcm=(img_orig_dcm*slope) + intercept

            if (0x28, 0x1050) in dicom_file:
                WindowCenter = dicom_file["WindowCenter"].value
                if not isinstance(WindowCenter, float) :
                    WindowCenter = 40
            if (0x28, 0x1051) in dicom_file:
                WindowWidth = dicom_file["WindowWidth"].value
            
            
            arraytopng = zoom(img_modif_dcm, (1/4, 1/4))
            arraytopng = np.stack((arraytopng,)*3, axis=-1)
            volume_pour_label[k,:,:,:]=arraytopng
            
            del arraytopng
            
        volume_pour_label     = np.asarray(volume_pour_label, dtype=np.float16)   
        volume_pour_label,a,b = normalize(volume_pour_label)
        volume_pour_label     = WL_scaled(WindowCenter,WindowWidth,volume_pour_label,a,b)
            
        if verbose>1:affichage3D(volume_pour_label, 64, axis=2)

        if verbose >0 : print("Analyse du volume pour obtention d'un scanner abdominopelvien")
        if verbose >0 : AA=1 #Permet de mettre corriger le verbose si celui_ci était de 2.
        else :          AA=0
        AUTO_BATCH = int(Compute_capacity*5.3)
        prediction = model.predict(volume_pour_label, verbose =AA, batch_size=AUTO_BATCH)
        prediction0_1 = np.zeros_like(prediction, dtype=None, order='K', subok=True, shape=None)
        for i in range (0,np.shape(prediction)[0]):
            prediction0_1[i][np.argmax(prediction[i])] = 1
        
        moyenne = TESTINGPRED(prediction0_1,classtotest,"ProportionGlobale",verbose=0)
        if moyenne[4] <0.2:
            stopNow = True
            erreur = "Le scanner ne possède pas assez de coupes pelviennes"
        
        fin = TESTINGPRED(prediction0_1,classtotest,
                          "ProportionFinale",
                          nombredecoupes=50//ratioECHANTILLONAGE,
                          numerocoupeinitial = 0,verbose=0)
        if (fin[1]+fin[2]+fin[0]+fin[5]) >0.35 : #plus de 35 % finaux sont  cervical, diaphragme, abdo ou thorax
            stopNow = True
            erreur = "La fin du volume n'est pas un scanner abdominopelvien"

    
    
    if stopNow == True:
        volume_numpy = np.empty(0)
        if verbose>0:print("   Arrêt précoce de niveau 2. Les images ont été chargées partiellement puis arrêtées : ",erreur)
        perduSUP = "Arret"
        facteur = "niveau2"
        
        if verbose>1:print("Mise à jour du fichier csv :", csv_path)
        
        #Essai remplir csv meme pour les arrets
        values=[]
        for de2 in metadata:
            if de2 in ds_img1:
                if ds_img1[de2].VR == "SQ":
                    values = values + "sequence"
                elif ds_img1[de2].name != "Pixel Data":
                    _ds = str(ds_img1[de2].value)[:64]
                    raw_ds = _ds.replace('\n','__')
                    raw_ds = raw_ds.replace('\r','__')
                    raw_ds = raw_ds.replace('\t',"__")
                    raw_ds = raw_ds.replace(',',"__")
                    values.append(raw_ds)

        end = time.perf_counter()
        Timing = end-start
        
        dictMETADATAS = dict(zip(metadata, values))
        dictODIASP = {'Name' : NOMFICHIER, "Duree" : Timing, "Erreur" : erreur,'OriginalSlices' : len(list_files), 'Path' : rootdir, "Archive" : None}
        dict3 = {**dictMETADATAS , **dictODIASP}
        df=pandas.read_csv(csv_path, delimiter=",")
        modDfObj = df.append(dict3, ignore_index=True)
        modDfObj.to_csv(csv_path, index=False)
        
        return volume_numpy, perduSUP, perduBAS, facteur, None
    
    
    
    if stopNow == False:    
        if verbose==0: print("Chargement, cela peut être long ...")
        
        """
        Nous allons maintenant retirer les coupes initiales ou coupes finales si jamais elles n'appartiennent pas au volume abdopelv
        """
        
        total = len(prediction) 
        tranchedelecture = 30 #Lecture des 30 premieres coupes : 
        for i in range(0,total,tranchedelecture//ratioECHANTILLONAGE):
            debut = TESTINGPRED(prediction0_1,classtotest,
                                "ProportionInitiale",
                                nombredecoupes=tranchedelecture//ratioECHANTILLONAGE,
                                numerocoupeinitial=i,verbose=0)
            if debut[5]+debut[1] > (debut[0]+debut[2]+debut[4]) : # plus de thorax et cervical que abdopelv
                if verbose>1:print(" Sur les ", tranchedelecture, " premières coupes : proportion de crane,cervical :", debut[1], "thorax :", debut[5],"  diaphragme :", debut[2],"  abdo :", debut[0],"  pelv :", debut[4],"  mbinf :", debut[3])
                liste_fichiers= liste_fichiers[tranchedelecture:]
                perduSUP += tranchedelecture
                if verbose>0:print("Supression de ",tranchedelecture," coupes dont la majorité est du crane, cervical ou thorax.")
        if verbose>0 and perduSUP==0 :print("... Pas de coupes crane ni cervical ni thorax majoritaires initialement.")
        
        
        total = len(prediction) #mise à jour suite à la découpe faite juste au dessus
        tranchedelecture = 30 
        for i in range(0,total,tranchedelecture//ratioECHANTILLONAGE):
            fin = TESTINGPRED(prediction0_1,classtotest,
                                "ProportionFinale",
                                nombredecoupes=tranchedelecture//ratioECHANTILLONAGE,
                                numerocoupeinitial=i,verbose=0)
            if fin[3] > (fin[4]+fin[0]) : # plus de mb inf que pelvien ou abdo
                if verbose>1:print(" Sur les ", tranchedelecture, " dernières coupes : proportion de crane,cervical :", debut[1], "thorax :", debut[5],"  diaphragme :", debut[2],"  abdo :", debut[0],"  pelv :", debut[4],"  mbinf :", debut[3])
                #if verbose>1:print("Proportion de abdominopelvien:", debut[0])
                liste_fichiers= liste_fichiers[:-tranchedelecture]
                perduBAS += tranchedelecture
                if verbose>0:print("Supression de ",tranchedelecture," coupes finales dont la majorité est du membre inférieur.")
        if verbose>0 and perduBAS==0 :print("... Pas de coupes membres inférieurs majoritaires à la fin.")

        del volume_pour_label
        
        
        
        
        #Creation du volume representant le scanner dont on garde les coupes
        volume_numpy=np.zeros((len(liste_fichiers),x_dim,y_dim))
        slope=float(ds_img1[0x28,0x1053].value)
        intercept=float(ds_img1[0x28,0x1052].value)
        for k in range (0,len(liste_fichiers)):
            dicom_file = pydicom.read_file(liste_fichiers[k])
            img_orig_dcm = (dicom_file.pixel_array)
            img_modif_dcm=(img_orig_dcm*slope) + intercept
            img_modif_dcm= np.asarray(img_modif_dcm, dtype=np.float16) 
            volume_numpy[k,:,:]=img_modif_dcm #ecrit une ligne correspondant à l'image
        volume_numpy = np.asarray(volume_numpy, dtype=np.float16)
        
        if len(liste_fichiers)>384 : #Cette partie de la fonction permet de s'affranchir des inégalités dépaisseurs de coupes.
            facteur = 384/(len(liste_fichiers))
        nbcoupesfinal = int(len(liste_fichiers)*facteur)
        if facteur !=1 :
            if verbose>0:print(len(liste_fichiers)," coupes ont été chargées puis le volume est ramené à ", nbcoupesfinal, " coupes")
            
            #volume_numpy = zoom(volume_numpy, (facteur, 1, 1)) 
            #CUPY
            if CUPY == True:
                cp.cuda.Device(0).use()
                x_gpu_0 = cp.asarray(volume_numpy)
                x_gpu_0 = MAGIC.zoom(x_gpu_0, (facteur, 1, 1))
                volume_numpy = cp.asnumpy(x_gpu_0)
                x_gpu_0 = None
            else :
                volume_numpy = zoom(volume_numpy, (facteur, 1, 1))
            
        else : 
            if verbose>0: print(len(liste_fichiers), " coupes ont étés chargées")
    
            
        #Sauvegarde .npy
        if save != False:
            if verbose>0: print("Sauvegarde de "+NOMFICHIER+" ("+str(nbcoupesfinal)+" coupes) dans le dossier "+save)
            np.save(os.path.join(save,NOMFICHIER),volume_numpy)
        
        #Affichage
        if verbose>1: 
            print("...dont voici l'image sagittale centrale")
            volume_numpy = np.asarray(volume_numpy, dtype=np.float16)
            affichage3D(volume_numpy, int(x_dim//2), axis=2)    
        

        #Mise a jour du csv
        if verbose>1:print("Mise à jour du fichier csv :", csv_path)
        values=[]
        for de2 in metadata:
            if de2 in ds_img1:
                if ds_img1[de2].VR == "SQ":
                    values = values + "sequence"
                elif ds_img1[de2].name != "Pixel Data":
                    _ds = str(ds_img1[de2].value)[:64]
                    raw_ds = _ds.replace('\n','__')
                    raw_ds = raw_ds.replace('\r','__')
                    raw_ds = raw_ds.replace('\t',"__")
                    raw_ds = raw_ds.replace(',',"__")
                    values.append(raw_ds)

        end = time.perf_counter()
        Timing = end-start
                    
        dictMETADATAS = dict(zip(metadata, values))
        dictODIASP = {'Name' : NOMFICHIER, "Duree" : Timing,'OriginalSlices' : nbcoupes, 'DeletedSlices' : str(perduSUP)+r"+"+str(perduBAS),'Facteur' : facteur , 'Path' : rootdir, "Archive" : None}
        dict3 = {**dictMETADATAS , **dictODIASP}

        df=pandas.read_csv(csv_path, delimiter=",")
        modDfObj = df.append(dict3, ignore_index=True)

        modDfObj.to_csv(csv_path, index=False)
       
    return volume_numpy, perduSUP, perduBAS, facteur, NOMFICHIER


#___________________________________________________________________________________________
#___________________FONCTIONS POUR AFFICHAGE DES IMAGES_____________________________________
#___________________________________________________________________________________________

def ApplyWindowLevel (Global_Level,Global_Window,image):
    """
    Utilisée dans FindL3
    Les valeurs des voxels en DICOM sont entre -2000 et +4000, pour afficher une image en echelle de gris (255 possibilités de gris sur un oridnateur classique) il faut réduire les 6000 possibilités à 255. Cette fonction est nécessaire avant d'afficher une image mais fait perdre des données (passage de 16 bits à  8 bits par pixel).
    Obligatoire pour sauvegarder une image png ou jpg mais fait perdre de l'information !
    
    On redéfinit les valeurs des pixels selon une largeur de fenetre et un centre

    Parameters
    ----------
        - Global_Level : centre de la fenetre (en UH)
        - Global_Window : largeur de la fenetre (en UH)
        - image : image ou volume numpy chargé en mémoire
        
    Returns
    -------
        - image_ret : l'image ou le volume après réglage du contraste.
    
    Notes
    -----
    Ne fonctionne PAS si l'image a déjà été normalisée. Dans ce cas utiliser WL_scaled en fournissant a et b (obtenu par la fonction normalize).
    
    """
    li=Global_Level-(Global_Window/2)
    ls=Global_Level+(Global_Window/2)
    image_ret=np.clip(image, li, ls)
    image_ret=image_ret-li
    image_ret=image_ret/(ls-li)
    image_ret=image_ret*255
    return image_ret


def Norm0_1 (volume_array):
    """
    les scanners ont des voxels dont la valeur est négative, ce qui sera mal interprété pour une image, il faut donc normaliser entre 0 et 1. Cela permet notamment de les afficher sous formlat image apres un facteur de *255.
    """
    a,b,c=volume_array.min(),volume_array.max(),volume_array.mean()
    volume_array_scale=(volume_array-a)/(b-a)
    return volume_array_scale,a,b,c


def WL_scaled (Global_Level,Global_Window,array,a,b):
    """
    Idem que ApplyWindowLevel mais corrigé par les facteurs a et b qui correpsondent au min et max, 
    >>> à utiliser à la place de ApplyWindowLevel si on a utilisé Norm0_1 ou normalize
    

    Utilisée dans FindL3
    Les valeurs des voxels en DICOM sont entre -2000 et +4000, pour afficher une image en echelle de gris (255 possibilités de gris sur un oridnateur classique) il faut réduire les 6000 possibilités à 255. Cette fonction est nécessaire avant d'afficher une image mais fait perdre des données (passage de 16 bits à  8 bits par pixel).
    Obligatoire pour sauvegarder une image png ou jpg mais fait perdre de l'information !
    
    On redéfinit les valeurs des pixels selon une largeur de fenetre et un centre
    On sauvegarde les bornes initiales dans les variables a et b dans le cas où l'on veuille modifier le contraste après coup

    Parameters
    ----------
        - Global_Level : centre de la fenetre (en UH)
        - Global_Window : largeur de la fenetre (en UH)
        - array : image ou volume numpy chargé en mémoire
        - a : minimum en UH avant normalize
        - b : maximum en UH avant normalize
        
    Returns
    -------
        - image_ret : l'image ou le volume après réglage du contraste.
    
    Notes
    -----
    Ne fonctionne QUE si l'image a déjà été normalisée. 
    
    """
    li=Global_Level-(Global_Window/2)
    ls=Global_Level+(Global_Window/2)
    li=li/b
    ls=ls/b
    image_ret=np.clip(array, li, ls)
    image_ret=image_ret-li
    image_ret=image_ret/(ls-li)
    return image_ret




#___________________________________________________________________________________________
#___________________FONCTIONS POUR VALIDER LA SEGMENTATION__________________________________
#___________________________________________________________________________________________




def normalize (volume_array):
    """
    Utilisée dans FindL3
    Les valeurs des voxels en DICOM sont entre -2000 et +4000, pour limiter les calculs du réseau de neurones il est conseillé de diminuer ces valeurs entre -1 et 1.
    On sauvegarde les bornes initiales dans les variables a et b dans le cas où l'on veule modifier le contraste après coup

    Parameters
    ----------
        - volume_array : volume numpy chargé en mémoire
        
    Returns
    -------
        - volume_array : volume numpy chargé en mémoire
        - a : valeur minimum du volume avant la fonction
        - b : valeur maximum du volume avant la fonction
    
    Notes
    -----
    a et b peuvent être rentrés tels quels dans la fonction de réglage du contraste WL_scaled
    
    """
    a,b=np.float(volume_array.min()),np.float(volume_array.max())
    volume_array = volume_array.astype(np.float)
    if abs(a)>abs(b) :
        c = abs(a)
    else:
        c= abs(b)
    if c != 0:
        volume_array_scale=volume_array/c
    return volume_array_scale,a,b


def axial_to_sag (volume_array, sens=1):
    """
    Utilisée dans FindL3
    rotation pour passer le volume de axial à sagittal

    Parameters
    ----------
        - volume_array : volume numpy chargé en mémoire
        - sens : int, 0 ou 1
    
    """
    volume_array = np.rot90(volume_array,k=sens,axes=(0,1))
    volume_array = np.rot90(volume_array,k=sens,axes=(2,0))
    return volume_array


def affichage3D(volume, k, axis=0):
    """
    affiche la coupe numéro k d'un volume, selon son axe axis

    Parameters
    ----------
        - volume : volume numpy chargé en mémoire
        - k : int, numéro de coupe
        - axis : int, 0 : axial ; 1 : coronal ; 2 : sag (dans le cas d'un volume chargé en axial)
    
    """
    f = plt.figure()
    if axis == 0:
        image1 = volume[k,:,:]
    if axis == 1:
        image1 = volume[:,k,:]
    if axis == 2:
        image1 = volume[:,:,k]
    plt.imshow(image1,cmap='gray')
    plt.show()
    return
    
    
def affichage2D(volume):
    """
    affiche un plan numpy 2D
    
    Parameters
    ----------
        - volume : plan numpy chargé en mémoire, en 2 dimensions
    
    """
    f = plt.figure()
    image1 = volume
    plt.imshow(image1,cmap='gray')
    plt.show()
    return


def AffichageMulti(volume, frequence, axis=0, FIGSIZE = 40):
    """
    affiche toutes les coupes d'un volume selon l'axe axis, avec une frequence entre les coupes définie
    
    Parameters
    ----------
        - volume : volume numpy chargé en mémoire
        - frequence : int, espace inter coupe (en voxels)
        - axis : int, 0 : axial ; 1 : coronal ; 2 : sag (dans le cas d'un volume chargé en axial)
        - FIGSIZE : taille des images pour l'affichage.
    
    """
    coupes = np.shape(volume)[axis]
    nb_images = coupes // frequence
    fig=plt.figure(figsize=(FIGSIZE, FIGSIZE))
    columns = 6
    if nb_images % columns >0 :
        rows = (nb_images // columns)+1
    else :
        rows = nb_images // columns
    for i in range(nb_images):
        i+=1
        fig.add_subplot(rows, columns, i)
        dix = frequence * i
        if axis == 0:
            plt.imshow(volume[dix,:,:], cmap='gray')
        elif axis == 1:
            plt.imshow(volume[:,dix,:], cmap='gray')
        elif axis == 2:
            plt.imshow(volume[:,:,dix], cmap='gray')
    plt.show(block=True)
    return
    


#___________________________________________________________________________________________
#___________________FONCTIONS POUR UTILISER LE RESEAU SUR UN VOLUME INTACT__________________
#___________________________________________________________________________________________

def NPY_to_DICOM (numpy=None, 
                  mode="name", 
                  csvpath=None, 
                  dossier = "L3", 
                  dirgeneral = r"C:\Users\alexa\OneDrive\Documents\ODIASP",
                  Center = 40,
                  Width = 400,
                  numerocoupe = None):
    """
    ---DEPRECATED---
    
    Cette fonction sert à créer un fichier dicom à partir des infos contenues dans le CSV et en entrant un volume numpy en entrée.
    On peut l'utiliser seule ou automatiquement dans la fonction FindL3
    a noter : 
    

    Parameters
    ----------
        - numpy : nom du numpy selon le csv. Optionnel si mode dossier.
        - mode : "name" ou "dossier :
            - en mode "name" : il faut nourrir l'argument "numpy" avec un string correspondant au nom d'un fichier .npy situé dans le 
                dossier "dossier"
            - en mode "dossier" : l'arg "numpy" ne sert à rien, la fonction va scanner tout le dossier "dossier" et créer un fichier 
                dicom pour chaque numpy trouvé
        - csv_path : chemin (complet) vers le csv où ont été enregistrées les metadatas de All-in-one
        - dossier : sous-dossier dans lequel se trouvent les numpy (si mode "dossier")
        - dirgeneral : Dossier de travail où ce situent les autres sous-dossier
        - Center = 40 et Width = 400 : ne pas toucher. Correspondent aux réglages de contraste pour l'image de sortie
        - numerocoupe : rentre l'information du numero de coupe
        
    Notes
    -----
    En l'état cette fonction n'est pas utilisée par FindL3 dans All-in-one : elle n'est pas nécessaire car nous récupérons directement le fichier dicom d'origine.
    Cette fonction pouvant s'avérer utile par ailleurs, nous la laissons donc en l'état.
    """
    
    if mode=="name" :
        if numpy==None:
            raise Error
        PATHduNUMPY = os.path.join(os.path.join(dirgeneral,dossier),numpy)
        VOLUME = np.load(PATHduNUMPY)
        image2d = VOLUME.astype(np.uint16)
        
        df=pandas.read_csv(csvpath, delimiter=",")
        df.set_index("Name", inplace=True)
        
        #Setting file meta information...
        meta = pydicom.Dataset()
        meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage  #<<<<<<<<<<<<<<<<<<<<<<<<<<<
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  

        #remplissage des metadonnées du dicom
        ds = Dataset()
        ds.file_meta = meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
        ds.PatientName = "Test^Firstname"
        ds.PatientID = "123456"
        ds.Modality = "CT"
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SamplesPerPixel = 1
        ds.HighBit = 15
        ds.ImagesInAcquisition = "1"
        ds.Rows = image2d.shape[0]
        ds.Columns = image2d.shape[1]
        ds.InstanceNumber = 1
        ds.ImagePositionPatient = r"0\0\1"
        ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
        ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
        ds.RescaleIntercept = "0" 
        ds.RescaleSlope = "1"
        if numerocoupe != None:
            ds.InstanceNumber = int(numerocoupe)
        #TaillePixel = str(df.at[numpy,"PixelSpacing"])[3:...] #DEBUG #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #ds.PixelSpacing = str(TaillePixel) + "\\" +str(TaillePixel) #DEBUG
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 1
        ds.WindowCenter = Center
        ds.WindowWidth = Width
        pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
        print("Setting pixel data... for ", numpy)
        ds.PixelData = image2d.tobytes()
        #enregistrement dans le dossier
        os.chdir(os.path.join(dirgeneral,dossier))
        ds.save_as(r"{0}.dcm".format(str(numpy)[0:-4]))
    
    if mode=="dossier":
        listfiles = os.listdir(os.path.join(dirgeneral,dossier))
        for item in listfiles :
            if str(item)[-4:] == ".npy":
                NPY_to_DICOM (item, mode="name", csvpath=csvpath, dossier=dossier, dirgeneral=dirgeneral)

                

#___________________________________________________________________________________________
#___________________FONCTIONS POUR LA VERSION DATA AUGMENT DU RESEAU _______________________
#___________________________________________________________________________________________


def Norm_and_Scale_andCrop(VOLUME,downsample = 0.5,CUPY=True):
    """
    prend un volume intact et commence à le traiter
    permet de diminuer la taille des fichiers
    """
    if CUPY== True:
        cp.cuda.Device(0).use()


        VOLUME = axial_to_sag(VOLUME)
        #volume_array_gpu = cp.asarray(VOLUME)
        #volume_array_gpu = cp.rot90(volume_array_gpu,1,axes=(0,1))
        #volume_array_gpu = cp.rot90(volume_array_gpu,1,axes=(2,0))

        hauteur = np.shape(VOLUME)[1]
        correction =("ok",1,0)
        if hauteur<384:
            ratio =384/hauteur
            volume_array_gpu = cp.asarray(VOLUME)
            volume_array_gpu = MAGIC.zoom(volume_array_gpu, (1, ratio, 1))
            VOLUME = cp.asnumpy(volume_array_gpu)
            volume_array_gpu = None
            correction = ("trop petit",hauteur, ratio)
        if hauteur>384: #a noter que ceci n'est pas censé arriver, la fonction d'import limitant la taille a 384 !
            VOLUME = VOLUME[:,-384:,:]
            delta = hauteur-384.
            correction = ("trop grand", hauteur, delta)

        VOLUME   = VOLUME[170:342,:,96:-32]
        if downsample != 1 :
            volume_array_gpu = cp.asarray(VOLUME)
            volume_array_gpu = MAGIC.zoom(volume_array_gpu, (1, downsample, downsample))
            VOLUME = cp.asnumpy(volume_array_gpu)
            volume_array_gpu = None


        VOLUMEnorm,a,b = normalize(VOLUME)
    

    #Version CPU
    else:
        VOLUME = axial_to_sag(VOLUME)
        hauteur = np.shape(VOLUME)[1]
        correction =("ok",1,0)
        if hauteur<384:
            ratio =384/hauteur
            VOLUME = zoom(VOLUME, (1, ratio, 1))
            correction = ("trop petit",hauteur, ratio)
        if hauteur>384: #a noter que ceci n'est pas censé arriver, la fonction d'import limitant la taille a 384 !
            VOLUME = VOLUME[:,-384:,:]
            delta = hauteur-384.
            correction = ("trop grand", hauteur, delta)

        VOLUME,a,b = normalize(VOLUME)

        VOLUME   = VOLUME[170:342,:,96:-32]
        if downsample != 1 :
            VOLUME = zoom(VOLUME, (1, downsample, downsample))

    return VOLUMEnorm, correction,a,b, VOLUME


def FindCoupeMediane(volume, verbose =0):
    x=0
    while np.sum(volume[:,x:,:]) > np.sum(volume[:,:x,:]):
        x+=1
    if verbose >0 : affichage2D(volume[:,x,:])
    if verbose >0 : affichage3D(volume, int(np.shape(volume)[0]/2), axis=0)
    return x

#________________________________________________________________________________________


def Find_L3 (name,
             model,
             NUMPY = "from_hardrive",
             downsample =0.5,
             csv_path = False,
             dirgeneral = r"C:\\Users\\alexa\\OneDrive\\Documents\\ODIASP",
             level = 40, window = 400, #
             savepng=False,
             savedcm=False, 
             nombredecoupesperduesSUP = 0, nombredecoupesperduesBAS = 0,
             facteurAgrandissement = 1,
             verbose = 2,
             Compute_capacity = COMPUTE_CAPACITY,
             CUPY=True
            ):
    """
    Cette fonction prend un volume numpy et le rentre dans le réseau de neurones.
    On peut appeler ce volume de deux facons : 
    - s'il est sauvegardé en rentrant son nom et en laissant NUMPY= "from_hardrive"
    - s'il est en mémoire en donnant le nom voulu pour name (pas d'importance hormis pour la cohérence du .csv) et en nourrisant le volume numpy dans l'argument NUMPY.

    Parameters
    ----------
        - name : nom donné au volume (doit correspondre au csv)
        - model : le model unet pour la segmentation de L3.
        - NUMPY = "from_hardrive",  par défaut le volume est chargé à partir des dossiers, sinon il suffit de nourrir ici un volumenumpy chargé en mémoire (c'est le cas dans la fonction all-in-one)
        - csv_path : chemin (complet) vers le csv où ont été enregistrées les metadatas de All-in-one
        - downsample : le downscaling qui a été utilisé pour le reseau de neurones. laisser à 0.5 si vous utilisez le reseau fourni.
        - dirgeneral : Dossier de travail où ce situent les autres sous-dossier
        - level = 40 et window = 400 : ne pas toucher. Correspondent aux réglages de contraste pour l'image de sortie
        - nombredecoupesperduesSUP = 0, nombredecoupesperduesBAS = 0 : nombres de coupes NON chargées, utilisés pour calculer la position sur le scanner initial.
        - facteurAgrandissement = 1 : zoom réalisé sur le volume chargé en amont.
        - verbose : qt de verbose. 0 pas de verbose ; 1 : texte seulement ; 2 : texte et images (met le terminal en pause à chaque image)
        - Compute_capacity = COMPUTE_CAPACITY  : pour adapter automatiquement le calcul réalisé par le reseau en fonction de la capacité de l'ordinateur
        
    Returns
    -------
    image  : un volume numpy sans perte d'information correspondant à l'axial de L3
    image_wl : une image, prête à être affichée (contraste pré-réglé mais avec perte d'information
    positionrelle : les coordonnés selon l'axe z du centre de L3 (sur le scanner entier, incluant les coupes non chargées)
    Par ailleurs, enregistre dans le csv les resultats et dans le dossier de sortie savepng les résultats en image png et savedcm en dicom
    
    Notes
    -----
    La segmentation musculaire est accessoire à cette étape :
        - pro : permet d'afficher les résultats à la volée, la sauvegarde des images intermédiaires (image sagittale etc) est accessoire
        - con : plus lent que de segmenter par la suite avec ODIASP.PredictMUSCLES
    
    """
    start = time.perf_counter()
    
    #Verification que le dossier de sortie existe
    if savepng != False :
        savepng  = DirVerification (savepng, DossierProjet=dirgeneral,verbose = 0)
    if savedcm != False :
        savedcm  = DirVerification (savedcm, DossierProjet=dirgeneral,verbose = 0)
    
    if type(NUMPY) == str :
        if type(NUMPY) == "from_hardrive" :
            NUMPY = Reading_Hardrive (name, Class="Images")
        else :
            NUMPY = NUMPY
    else :
        NUMPY = NUMPY

    #Traitement du volume pour qu'il soit accepté par le reseau de neurones.
    if verbose>0:print("Adaptation du volume avant recherche de L3")
    Model_import, correction,a,b, backup = Norm_and_Scale_andCrop(NUMPY,downsample=downsample,CUPY=CUPY) #enregistre la 'correction' réalisée dans le scaling
        
    backup = backup + abs(np.min(backup))
    backup = (backup/np.max(backup))*255
    backup= backup.astype(np.uint8)
    versionModel = Image.fromarray(backup[int(np.shape(backup)[0]/2),:,:])#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    
    Model_import = Model_import[:,:,:,np.newaxis]
    if verbose>0:a=1 #au cas où verbose = 2 ce qui ne serait pas accepté par la Method predict
    else:a=0
    if verbose>0:print("Localisation de L3...")
    AUTO_BATCH = int(Compute_capacity*1.3)
    prediction = model.predict(Model_import, verbose =a, batch_size=AUTO_BATCH)

    del Model_import
    
    #calcul du centre de gravité du volume donné au réseau pour obtenir le centre de L3 (et correction de sa valeur pour correspondre au volume numpy donné en entrée
    center = scipy.ndimage.center_of_mass(prediction, labels=None, index=None)#normal
    
    #center_median = FindCoupeMediane(prediction) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #prediction[prediction<(np.max(prediction)/2)]=0 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #center_bary_threshold = scipy.ndimage.center_of_mass(prediction, labels=None, index=None) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #center_median_threshold = FindCoupeMediane(prediction) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    
    upsample=1/downsample
    position = center[1]*upsample
    #center_median *= upsample#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #center_bary_threshold = center_bary_threshold[1] * upsample#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #center_median_threshold *= upsample#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    if correction[0]=="trop petit": #lit la correction
        position=int(position/correction[2])
        #center_median = int(center_median/correction[2]) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #center_bary_threshold = int(center_bary_threshold/correction[2]) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #center_median_threshold = int(center_median_threshold/correction[2])#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        
    elif correction[0]=="trop grand":
        position=int(position+correction[2])
        #center_median = int(center_median+correction[2]) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #center_bary_threshold = int(center_bary_threshold+correction[2]) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #center_median_threshold = int(center_median_threshold+correction[2])#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        
        
    else :
        position = int(position)
    if verbose>0:print("Axial position : "+str(position), "dans ce volume")
    positionreelle = int((position*(1/facteurAgrandissement)) +nombredecoupesperduesSUP)
    #center_median = int((center_median*(1/facteurAgrandissement)) +nombredecoupesperduesSUP) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #center_bary_threshold = int((center_bary_threshold*(1/facteurAgrandissement)) +nombredecoupesperduesSUP) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #center_median_threshold = int((center_median_threshold*(1/facteurAgrandissement)) +nombredecoupesperduesSUP)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    
    
    if verbose>0 and positionreelle != position : print("Axial position : ",positionreelle, " dans le volume initial")
    #Standard_deviation = np.std(prediction)
    #Certitude = (Standard_deviation*100)**6
    #if verbose>0:print("Estimation de la confiance : ", Certitude)
    
    NUMPY = np.asarray(NUMPY, dtype=np.float16)
    image = NUMPY[position,:,:] #image axiale centrée sur le baricentre
    image_wl=ApplyWindowLevel(level,window,image) #réglages de contraste
    sagittal = NUMPY[:,:,int(center[0])+170] #image sag centrée sur le baricentre  #sagittal = NUMPY[:,:,int(center[0])+128] #image sag centrée sur le baricentre 
    sagittal_wl=ApplyWindowLevel(level,window,sagittal) #réglages de contraste
    sagittal_wl[position,:] = np.amax(NUMPY)/6 #crée la ligne horizontale sur l'image sag montrant la coupe axiale trouvée 

    sagittalPRED = prediction[int(center[0]),:,:]
    sagittalPRED = sagittalPRED[:,:,0]

    if correction[2] == 0:factueurdecorrection=1
    else : factueurdecorrection=correction[2]
    sagittalPRED = zoom(sagittalPRED, (1/(factueurdecorrection*downsample),1/downsample))
    sagittalPRED *=255
    mask_a_afficher =np.zeros(np.shape(sagittal_wl))
    mask_a_afficher[:,96:-32] = sagittalPRED
    
    mask_a_save = mask_a_afficher/np.max(mask_a_afficher)
    
    #Gestion des problèmes de nom de fichier vs nom de dossier
    if str(name)[-4:] == r".npy":
        nameNPY=str(name)[-43:]
        name__=str(name)[-43:-4]
    else:
        name__=str(name)[-39:]
        nameNPY=str(name)[-39:]+r".npy"
        
        
    if savepng != False:       
        #saving the axial image
        arraytopng,_,_,_ = Norm0_1(image_wl)
        arraytopng *=255
        arraytopng= arraytopng.astype(np.uint8)
        im = Image.fromarray(arraytopng)
        im.save(os.path.join(savepng,name__)+r"_axial.png")
        versionModel.save(os.path.join(savepng,name__)+r"VersionScaleModel.png") #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        #saving the sagittal image
        sagtopng,_,_,_ = Norm0_1(sagittal_wl+mask_a_save/6)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        sagtopng *=255
        sagtopng= sagtopng.astype(np.uint8)
        im2 = Image.fromarray(sagtopng)
        im2.save(os.path.join(savepng,name__)+r"_sag.png")
        
    if savedcm != False: 
        #saving the dicom
        NPY_to_DICOM (numpy=nameNPY, mode="name", csvpath=csv_path, dossier = savedcm, dirgeneral = dirgeneral,numerocoupe = positionreelle) 
        
    if verbose>1:
        _, ax = plt.subplots(1,2,figsize=(25,25))
        ax[0].imshow(image_wl, cmap='gray')
        ax[1].imshow(sagittal_wl, cmap='gray')
        ax[1].imshow(mask_a_afficher,  cmap='magma', alpha=0.5)
        plt.show()
        
    if csv_path != False: #mettre a jour le csv avec pandas
        end = time.perf_counter()
        Timing = end-start
        
        df=pandas.read_csv(csv_path, delimiter=",")
        df.set_index("Name", inplace=True)
        df.at[nameNPY,"L3Position"] = position
        #df.at[nameNPY,"Standard_deviation"] = Standard_deviation
        #df.at[nameNPY,"Certitude"] = Certitude
        df.at[nameNPY,"L3Original"] = positionreelle
        #df.at[nameNPY,"center_median"] = center_median
        #df.at[nameNPY,"center_bary_threshold"] = center_bary_threshold
        #df.at[nameNPY,"center_median_threshold"] = center_median_threshold
        df.at[nameNPY,"DureeL3"] = Timing
        df.to_csv(csv_path)

    return image, image_wl, positionreelle



def All_in_One(dossierDICOM,
               METADATAS,
               csv_path,
               MODEL_niveau_de_coupe,
               Model_segmentation_L3,
               Model_segmentation_muscles = None,
               DIR_SORTIE = False, 
               VERBOSE = 2,
               WINDOW_CENTER = 40,
               WINDOW_WIDTH = 400,
               DossierDeTravail = None,
               CUPY = True
              ):
    """
    Charge les examens depuis un dossier, trouve les images correspondant au scan abdopelvien puis segmente ce volume pour trouver L3.
    On peut segmenter les muscles dans la même étape pour afficher le resultat en une seule fois mais ceci nécessite de charger le réseau de segmentation musculaire à chaque fois : cette méthode est plus consommatrice en ressources.
    il est conseillé de ne pas segmenter les muscles à cette étape mais de le faire en une seule fois par la suite.
    
    Parameters
    ----------
        - dossierDICOM : Il s'agit du dossier d'import où se situent les images
        - METADATAS : les informations que l'on veut sauvegarder parmi les métadonnées des DICOM
        - csv_path : chemin (complet) vers le csv où ont été enregistrés, les metadatas de All-in-one
        - MODEL_niveau_de_coupe : le modele de labelisation
        - Model_segmentation_L3 : le modele de semgentation de L3
        - Model_segmentation_muscles : le model (ODIASP) pour la segmentation musculaire,  non obligatoire
        - DIR_SORTIE : Dossier dans lequel seront sauvegardés les images de résultats (les résultats sont de toute facon enregistrés en format texte dans le csv
        - VERBOSE : qt de verbose. 0 pas de verbose ; 1 : texte seulement ; 2 : texte et images (met le terminal en pause à chaque image)
        - WINDOW_CENTER = 40 et WINDOW_WIDTH = 400 : ne pas toucher.
        - DossierDeTravail : Dossier où seront enregistrées les infos, nécessaire si on utilise la segmentation musculaire par la suite. Optionnel si la segmentation musculaire est faire à la volée.
        
    Returns
    -------
    Rien.
    Mais enregistre dans le csv les resultats et dans le dossier de sortie DIR_SORTIE les résultats
    
    Notes
    -----
    La segmentation musculaire est accessoire à cette étape :
        - pro : permet d'afficher les résultats à la volée, la sauvegarde des images intermédiaires (image sagittale etc) est accessoire
        - con : plus lent que de segmenter par la suite avec ODIASP.PredictMUSCLES
    
    """
    #la fonction fast_scandir est récursive et crée une liste des sous dossiers que l'on trouve dans le DossierGeneral
    dir_to_scan = fast_scandir(dossierDICOM)
    
    nb_niv1 = 0
    nb_niv2 = 0
    i=1
    for dirs in dir_to_scan:
        print ("Chargement du dossier ", i,r"/",len(dir_to_scan)," : " + str(dirs))
        i+=1
        """
        Import du scanner, labelisation avec le premier reseau et mise a jour du csv
        """
        start_time = time.clock() #TIME


        volume_numpy, perduSUP, perduBAS, facteur, NOM = import_dicom_to_abdopelv(dirs,
                                                                                  metadata = METADATAS,
                                                                                  csv_path = csv_path,
                                                                                  save= False, 
                                                                                  model = MODEL_niveau_de_coupe,
                                                                                  verbose = VERBOSE,
                                                                                  CUPY=CUPY)
        
        finImport_time = time.clock() #TIME
        print(finImport_time - start_time, "secondes pour l'import")#TIME
        
        if perduSUP == "Arret" : #export_dicomDir_to_numpyV2 renvoit la variable perdu = "Arret" si jamais elle s'est arrêtée seule
            if facteur == "niveau1" :
                nb_niv1 +=1
            if facteur == "niveau2" :
                nb_niv2 +=1
            if VERBOSE >1 : print ("\n \n")

        

        else :
            """
            Transmission pour prediction de L3
            """
            image, image_wl, position = Find_L3 (NOM, 
                                                 model = Model_segmentation_L3,
                                                 NUMPY = volume_numpy,  
                                                 downsample = 0.5,
                                                 csv_path = csv_path,
                                                 dirgeneral = DossierDeTravail,
                                                 level = WINDOW_CENTER, window = WINDOW_WIDTH,
                                                 savepng=DIR_SORTIE,
                                                 savedcm=False,
                                                 nombredecoupesperduesSUP = perduSUP,nombredecoupesperduesBAS = perduBAS,
                                                 facteurAgrandissement = facteur,
                                                 verbose = VERBOSE,
                                                 CUPY=CUPY)
            
            
            finL3 = time.clock() #TIME
            print(finL3 - finImport_time, "secondes pour la segmentation L3 ")#TIME
            
            """
            On recupere la postion de L3 pour aller chercher le fichier dicom d'origine qui lui correspond
            """
            #il faut les trier dans l'ordre de la sequence de scanner (ce qui n'est pas l'ordre alphabetique du nom des fichiers)
            inter = {} 
            list_files = os.listdir(dirs)
            for f in list_files:
                if not os.path.isdir(f):
                    f_long = os.path.join(dirs, f)
                    _ds_ = pydicom.dcmread(f_long,specific_tags =["ImagePositionPatient","SliceThickness"])
                    inter[f_long]=_ds_.ImagePositionPatient[2]
            inter_sorted=sorted(inter.items(), key=lambda x: x[1], reverse=True) 
            liste_fichiers=[x[0] for x in inter_sorted]

            dicom_file = pydicom.dcmread(liste_fichiers[position], force=True)
            dicom_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            img_orig_dcm = (dicom_file.pixel_array)
            slope=float(dicom_file[0x28,0x1053].value)
            intercept=float(dicom_file[0x28,0x1052].value)
            img_modif_dcm=(img_orig_dcm*slope) + intercept
            img_dcm_wl=ApplyWindowLevel(WINDOW_CENTER,WINDOW_WIDTH,img_modif_dcm) 



            if DIR_SORTIE != False :
                DIR_SORTIE  = DirVerification (DIR_SORTIE, DossierProjet=DossierDeTravail,verbose = 0)
                name = str(NOM)[:-4]
                name += "_" + str(liste_fichiers[position])[-8:]
                namepng = name + "_FichierOrigine.png"
                namedcm = name + "_FichierOrigine.dcm"
                SAVEPATH = os.path.join(DIR_SORTIE,namepng)
                im2 = Image.fromarray(img_dcm_wl)
                im2 = im2.convert("L")
                im2.save(SAVEPATH)
                copy2(liste_fichiers[position], os.path.join(DIR_SORTIE,namedcm))

            if Model_segmentation_muscles != None :
                """
                On teste notre coupe L3 pour segmenter le muscle automatiquement
                """
                if VERBOSE >0 :a=1
                else: a=0
                image_wl = image_wl[np.newaxis,:,:,np.newaxis]
                imagepourreseau = image_wl/255
                SEGMuscles = Model_segmentation_muscles.predict(imagepourreseau, verbose=a)


                """
                Calcul de la surface segmentée
                """
                pixelspacing=float(str(dicom_file[0x28,0x0030].value)[1:7])

                mask = copy(img_modif_dcm) #Creation d'un masque pour ne garder que les pixels entre 29 et 150UH (cf litterature)
                mask[mask > 150] = -1000
                mask[mask >= -29] = 1
                mask[mask < -29] = 0
                SEGMuscles_masked = copy(SEGMuscles[0,:,:,0])
                SEGMuscles_masked[SEGMuscles_masked <= 0.5] = 0
                SEGMuscles_masked[SEGMuscles_masked > 0.5] = 1
                SEGMuscles_masked = np.multiply(SEGMuscles_masked,mask)
                surface_0 = np.sum(SEGMuscles_masked)

                if VERBOSE >1 :
                    _, ax = plt.subplots(1,3,figsize=(25,25))
                    ax[0].imshow(image_wl[0,:,:,0], cmap='gray') #L'image provenant du numpy pour le calcul
                    ax[1].imshow(img_dcm_wl, cmap='gray')#L'image provenant du dicom chargé à nouveau
                    ax[1].imshow(SEGMuscles[0,:,:,0], cmap='magma', alpha=0.5)
                    ax[2].imshow(SEGMuscles_masked, cmap='Reds')
                    plt.show()


                if DIR_SORTIE != False :
                    namemask = name + "_Mask.png"
                    SAVEPATHmask = os.path.join(DIR_SORTIE,namemask) 
                    SEGMuscles_masked *=255
                    SEGMuscles_masked= SEGMuscles_masked.astype(np.uint8)
                    im_mask = Image.fromarray(SEGMuscles_masked)
                    im_mask.save(SAVEPATHmask)



                if csv_path != False: #mettre a jour le csv avec pandas
                    df=pandas.read_csv(csv_path, delimiter=",")
                    df.set_index("Name", inplace=True)
                    df.at[NOM,"Surface"] = float("{0:.2f}".format(surface_0*(pixelspacing**2)/100))
                    df.to_csv(csv_path)
                    if VERBOSE >1 : print(df.loc[NOM])
                        
                finFonction= time.clock() #TIME
                print(finFonction - finL3, "secondes pour le reste ")#TIME
                print(finFonction - start_time, "secondes pour la totalité")#TIME

            if VERBOSE >0 :print("\n")

    print("Fin : ",len(dir_to_scan)," dossiers analysés dont \n -", nb_niv1, "arrêts de niveau1 \n -",nb_niv2, "arrêts de niveau2." )
    return


def load_modelsODIASP(model_directory, strategy=None, label = True, L3 = True, muscle=True, verification = True):
    
    """
    Loads previously trained models from a single provided directory folder 
    
    Parameters
    ----------
        - model_directory : string directing to model directory, directory of the folder containing all trained models
        - strategy : la strategie utilisée (voir la doc de tensorflow)
        - label, L3, muscle : décide quels modèles on veut charger.
        
    Returns
    -------
        - modelLABEL : le model pour la labelisation des images
        - modelL3 : le model pour la segmentation de L3
        - modelMUSCLE : le model pour la segmentation des muscles
    
    Notes
    -----
    Ensure model filenames remain unchanged
    
    """
    modelLABEL = None
    modelL3 = None
    modelMUSCLE = None
    modelVerif = None
    
    def dice_coef(y_true, y_pred):
        '''
        Metric
        '''
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score


    def dice_loss(y_true, y_pred):
        '''
        Loss function
        '''
        loss = 1 - dice_coef(y_true, y_pred)
        return loss


    def bce_dice_loss(y_true, y_pred):
        '''
        Mixed crossentropy and dice loss.
        '''
        loss = keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
        return loss

    print("\nChargement des différents réseaux ODIASP")
    if strategy != None :
        with strategy.scope():
            if label == True : 
                modelLABEL  = tf.keras.models.load_model(os.path.join(model_directory, "modelLABEL"))
            if L3 == True :
                modelL3     = tf.keras.models.load_model(os.path.join(model_directory, "modelL3")) 
            if muscle== True :
                modelMUSCLE = tf.keras.models.load_model(os.path.join(model_directory, "modelMUSCLE"), compile=False) #get_unet_512_V4_30ep
                modelMUSCLE.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coef])
            if verification == True :
                modelVerif  = tf.keras.models.load_model(os.path.join(model_directory, "modelVerification")) 
    
    else : 
        if label == True : 
            modelLABEL  = tf.keras.models.load_model(os.path.join(model_directory, "modelLABEL"))
        if L3 == True :
            modelL3     = tf.keras.models.load_model(os.path.join(model_directory, "modelL3")) 
        if muscle== True :
            modelMUSCLE = tf.keras.models.load_model(os.path.join(model_directory, "modelMUSCLE"), compile=False) #get_unet_512_V4_30ep
            modelMUSCLE.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coef])
        if verification == True :
                modelVerif  = tf.keras.models.load_model(os.path.join(model_directory, "modelVerification")) 
    
    return modelLABEL, modelL3, modelMUSCLE, modelVerif

    
    
def PredictMUSCLES(DossierImport,
                   csv_path,
                   Model_segmentation_muscles,
                   Model_Verification,
                   DIR_SORTIE = False,
                   VERBOSE = 2,
                   WINDOW_CENTER = 40, WINDOW_WIDTH = 400,
                    ):
    """
    Prediction des muscles à partir d'un dossier contenant tous les fichiers DICOM
    
    Parameters
    ----------
        - DossierImport : Il s'agit du dossier d'import o'u se situent les images
        - csv_path : chemin (complet) vers le csv où ont été enregistrés, les metadatas de All-in-one
        - Model_segmentation_muscles : le model (ODIASP) pour la segmentation musculaire
        - DIR_SORTIE : Dossier dans lequel seront sauvegardés les images de résultats (les résultats sont de toute facon enregistrés en format texte dans le csv
        - VERBOSE : qt de verbose
        - WINDOW_CENTER = 40 et WINDOW_WIDTH = 400 : ne pas toucher.
        
    Returns
    -------
    Trois listes synchronisées sur le :
        - le nom du fichier
        - la largeur du pixel dans ce scanner
        - la surface segmentée par le reseau
    
    Notes
    -----
    Cette fonction  utilise le même reseau que pour All in one. Elle calcule toutes les segmentations de muscle en une seule fois cependant, une fois que All-in-one est terminé. Elle est donc plus rapide mais ne permet pas d'afficher les résultats en live.
    
    """
    touslesfichiers = os.listdir(DossierImport)
    fichiersdcm = []
    #fichiersPNG = []
    for item in touslesfichiers :
        if str(item)[-19:] == r"_FichierOrigine.dcm":
            fichiersdcm.append(item)
        #elif str(item)[-19:] == r"_FichierOrigine.png":
            #fichiersPNG.append(item)   
    del touslesfichiers
    
    volume=np.zeros((len(fichiersdcm),512,512))
    volume_wl=np.zeros((len(fichiersdcm),512,512))
    liste_pixelspace = []
    liste_surfacecm2 = []
    liste_surfaceGraissecm2 = []
    
    for k in range (0,len(fichiersdcm)):
        f_long = os.path.join(DossierImport, fichiersdcm[k])

        #recupère l'image du fichier DICOM
        dicom_file = pydicom.dcmread(f_long, force=True)
        img_orig_dcm = (dicom_file.pixel_array)
        slope=float(dicom_file[0x28,0x1053].value)
        intercept=float(dicom_file[0x28,0x1052].value)
        img_modif_dcm=(img_orig_dcm*slope) + intercept

        #ecrit une ligne correspondant à l'image sur le volume
        volume[k,:,:]=img_modif_dcm 
        
        #enregistre le pixelspacing
        pixelspacing=str(dicom_file[0x28,0x0030].value)
        pixelspacing = float(pixelspacing.split(",")[0][1:])
        liste_pixelspace.append(pixelspacing)
    
    
    #On adapte le volume pour le rendre lisible par le reseau
    for k in range (0,len(fichiersdcm)):
        volume_wl[k,:,:]=ApplyWindowLevel(WINDOW_CENTER,WINDOW_WIDTH,volume[k,:,:])
    volume_to_predict = volume_wl[:,:,:,np.newaxis]        
    
    #Creation d'un masque pour ne garder que les pixels entre -29 et 150UH (cf litterature)
    mask = copy(volume) 
    mask[mask > 150] = -1000
    mask[mask >= -29] = 1
    mask[mask < -29] = 0 
    mask = mask[:,:,:,np.newaxis]  
    
    #Obtention dela surface graisseuse grace aux pixels entre -190 et -29UH (cf litterature)
    graisse = copy(volume) 
    graisse[graisse >   -29] = -1000
    graisse[graisse >= -190] = 1
    graisse[graisse <  -190] = 0 
    
    #On le fait passer dans le reseau de neurones
    if VERBOSE >0 :a=1
    else: a=0
    volume_to_predict = volume_to_predict/255
    if VERBOSE >0 : print("Segmentation avec le reseau d'ODIASP")
    SEGMuscles = Model_segmentation_muscles.predict(volume_to_predict, verbose=a)
    del volume_to_predict
    del volume
    
    #reglage du threshold, abandonné, cf article
    #SEGMuscles[SEGMuscles <= 0.5] = 0   
    #SEGMuscles[SEGMuscles > 0.5] = 1
    
    SEGMuscles_masked = np.multiply(SEGMuscles,mask)
    
    #Calcul de la densité du muscle
    
    
    for image in range (0,len(fichiersdcm)):
        sommepixels = np.sum(SEGMuscles_masked[image,:,:,0])
        pixelspace = liste_pixelspace[image]
        surfacecm2 = sommepixels*(pixelspace**2)/100
        liste_surfacecm2.append(surfacecm2)
        
        sommeGraisse = np.sum(graisse[image,:,:])
        surfaceGraisse = sommeGraisse*(pixelspace**2)/100
        liste_surfaceGraissecm2.append(surfaceGraisse)
    
    #Creation de la carte affichant toutes les segmentations :
    resultat = np.ones_like(graisse)
    resultat -= graisse
    resultat -= SEGMuscles_masked[:,:,:,0]
    
    volume_wl = (volume_wl)* resultat
    volume_wl = np.stack((volume_wl,)*3, axis=-1)
    volume_wl[:,:,:,2]=graisse*255.
    volume_wl[:,:,:,0]=SEGMuscles_masked[:,:,:,0]*255.
    
    Verification_volume = volume_wl/255
    Verification_volume = zoom(Verification_volume, (1, .5, .5,1))
    if VERBOSE >0 : print("Vérification des résultats par le réseau de neurones dédié :")
    Predictions = Model_Verification.predict(Verification_volume, verbose=a)
    labels = ["erreurL3", "erreurMuscles", "Ok"]
    predictions = [labels[k] for k in np.argmax(Predictions,axis=1)]
    
    #Sauvegarde
    if DIR_SORTIE != False :
        for image in range (0,len(fichiersdcm)):
            namemask = str(fichiersdcm[image])[:-19] + "_Mask.png"
            SAVEPATHmask = os.path.join(DIR_SORTIE,namemask)
            image_du_masque = SEGMuscles_masked[image,:,:,0]
            image_du_masque *=255
            image_du_masque= image_du_masque.astype(np.uint8)
            im_mask = Image.fromarray(image_du_masque)
            im_mask.save(SAVEPATHmask)
    
            nameGraisse = str(fichiersdcm[image])[:-19] + "_Graisse.png"
            SAVEPATHgraisse = os.path.join(DIR_SORTIE,nameGraisse)
            image_de_la_graisse = graisse[image,:,:]
            image_de_la_graisse *=255
            image_de_la_graisse= image_de_la_graisse.astype(np.uint8)
            im_graisse = Image.fromarray(image_de_la_graisse)
            im_graisse.save(SAVEPATHgraisse)
            
            nameAffichage = str(fichiersdcm[image])[:-19] + "_Verification({}).png".format(predictions[image])
            SAVEPATHaffichage = os.path.join(DIR_SORTIE,nameAffichage)
            image_overlay = volume_wl[image,:,:,:]
            image_overlay= image_overlay.astype(np.uint8)
            im_overlay = Image.fromarray(image_overlay)
            im_overlay.save(SAVEPATHaffichage)
    
    if csv_path != False: #mettre a jour le csv avec pandas
        df=pandas.read_csv(csv_path, delimiter=",")
        df.set_index("Name", inplace=True)
        for k in range (0,len(fichiersdcm)) :
            Nom_en_numpy = str(fichiersdcm[k])[:-28] + r".npy"
            df.at[Nom_en_numpy,"Surface"] = float("{0:.2f}".format(liste_surfacecm2[k]))
            df.at[Nom_en_numpy,"SurfaceGraisseuse"] = float("{0:.2f}".format(liste_surfaceGraissecm2[k]))
            df.at[Nom_en_numpy,"PredictionVerification"] = predictions[k]
        df.to_csv(csv_path)
    
    return fichiersdcm, liste_pixelspace, liste_surfacecm2    
    

def archivage(origine,archives,verbose):
    
    """
    Deplace les examens déjà interprétés dans un dossier archive généré automatiquement selon l'heure d'archivage.
    
    Parameters
    ----------
    origine : lieu où se situent les examens à archiver
    archives : dossier danslequel créer le sous dossier d'archivage
    verbose : texte de renseignement si >0
    
    Notes
    -----
    Il est possible de supprimer les archives si l'on veut, dans le fonction normal du logiciel elles ne servent plus à rien.
    
    """
    date = datetime.datetime.now()
    destination = "Archive_" + date.ctime().replace(':', '-')
    path = os.path.join(archives, destination)
    
    if verbose>0: print("Archivage...")
    if os.path.isdir(path)==False:
        try:
            os.mkdir(path)
        except OSError:
            print ("Erreur dans la creation du dossier %s" % path)
        else:
            print ("Creation du dossier %s " % path)

    files = os.listdir(origine)
    with alive_bar(len(files)) as bar :
        for f in files:
            move(os.path.join(origine,f), path)
            bar()  
    return



def import_one_XLS(fichierexcel,dossier):
    #Import de l'excel en dataframe
    book = openpyxl.load_workbook(os.path.join(dossier,fichierexcel))
    ws = book['main results']
    data = ws.values
    columns = next(data)[0:]
    dfauto = pandas.DataFrame(data, columns=columns)

    #Recuperation des colonnes utiles uniquement  
    features_cols_auto = ["Patient ID", "Age", "Sex", "Muscle CSA", "IMAT CSA", "VAT CSA", "SAT CSA", "CT date","Muscle HU","Scan folder","Instance_UID"]
    dfauto = dfauto[features_cols_auto]

    #Changement des noms et types
    dfauto = dfauto.rename(columns={"Patient ID": "PatientID", "CT date": "StudyDate"})
    dfauto["PatientID"] = dfauto["PatientID"].astype('int')
    dfauto["StudyDate"] = dfauto["StudyDate"].astype('int')
    
    #Calcul du Muscle Graisse ratio
    #dfauto['MGratio'] = dfauto.apply(lambda row: row['Muscle CSA'] / (row['IMAT CSA']+row['VAT CSA']+row['SAT CSA']), axis=1)

    return dfauto


def import_all_XLS(dossier):
    listetotal = os.listdir(dossier)
    listeExcels = []
    for item in listetotal :
        if str(item)[-5:] == r".xlsx" and str(item)[:8] ==r"Results ":
            listeExcels.append(item)
    
    if len(listeExcels)>0 :
        result = import_one_XLS(listeExcels[0], dossier=dossier)
        for i in range(1,len(listeExcels)):
            dfauto = import_one_XLS(listeExcels[i], dossier=dossier)
            result = pandas.concat([result, dfauto])

        result.drop_duplicates(keep="first",inplace=True) 
    
    else :
        result=None
        
    return result, listeExcels


def RecuperationErreurs(dataframe, dossierExport, pathdexport, verbose = 1):
    """
    Suppression des lignes dans le xls dont ne veut pas après vérification du réseau de neurone
    
    Parameters
    ----------
        - dataframe : le dataframe provenant d'AutoMATiCA
        - dossierExport : chemin vers le dossier où sont enregistrés les résultats finaux
        - pathedexport, correspond à PATHdEXPORT, le dossier des resultats intermediaires
        
    Returns
    -------
    le dataframe avec les informations justes et le dataframe avec les informations fausses
    
    """
    #Creation d'une liste des examens erronnés d'apres le reseau de verification
    LISTEVERIF = os.listdir(dossierExport)
    ErreursL3 = []
    ErreursMuscles = []
    nbinitial = len(dataframe)
    for item in LISTEVERIF :
        if "erreurL3" in item :
            name = item[:-27]
            name=os.path.join(pathdexport,str(name)+r"_FichierOrigine.dcm")
            ErreursL3.append(name)
        elif "erreurMuscles" in item :
            name = item[:-32]
            name=os.path.join(pathdexport,str(name)+r"_FichierOrigine.dcm")
            ErreursMuscles.append(name)
    
    dataframe["PredictionVerification"] = "Ok"
    
    for erreur in ErreursL3 : 
        dataframe.loc[dataframe["Scan folder"]==erreur, 'PredictionVerification'] = "erreurL3"
    for erreur in ErreursMuscles : 
        dataframe.loc[dataframe["Scan folder"]==erreur, 'PredictionVerification'] = "erreurMuscles"
    
    #if len(ErreursL3) + len(ErreursMuscles) == 0:
    #    dataframe["PredictionVerification"] = "Ok"
    
    dfErreursAuto = dataframe[(dataframe['PredictionVerification'] == "erreurL3") | (dataframe['PredictionVerification'] == "erreurMuscles")]
    dataframe = dataframe.drop(dfErreursAuto.index)
        
    if verbose>0: print("Les {} résultats sont séparés en :\n - {} résultats corrects,\n - {} résultats douteux concernant la localisation en L3,\n - {} résultats douteux concernant la segmentation muscles et graisses.".format(nbinitial,len(dataframe),len(ErreursL3),len(ErreursMuscles)))    
    return dataframe, dfErreursAuto



def FusionResultats(dossier,csvpathtemp,resultatspath,DIR_SORTIE,DossierResultasInterm,patharchives=None,erreurautorisee = 0.25, verbose=1):
    """
    Fusion des tableurs de résultats 
    
    Parameters
    ----------
        - dossier : string, chemin vers le dossier principal où se situent les fichierx excels de résultats d'AutoMATiCA
        - csvpathtemp : string, chemin vers le .csv remplit lors de la localisation de L3
        - resultatspath : string, chemin vers le .csv contenant les résultats finaux
        - DIR_SORTIE : string, chemin vers le dossier où sont enregistrées les images finales
        - DossierResultasInterm : string, chemin vers le dossier où sont enregistrées les DICOM en L3
        - patharchives : string, optionnel, chemin vers le dossier d'archivage
        - erreurautorisee : float <0 et <1, optionnel, indique le pourcentage d'erreur autorisée entre notre segmentation et celle d'automatica 
        
    Returns
    -------
        - newresult : dataframe, les reponses considérées comme correctes
        - AllErrors : dataframe, les reponses considérées comme fausses
    
    Notes
    -----
    Sauvegarde automatiquement les .csv et .xlsx des résultats
    
    """
    pandas.set_option('mode.chained_assignment', None)
    
    #Chargement du csv (ODIASP)
    if os.path.isfile(csvpathtemp)==True:
        df = readCSV(csvpathtemp) 
        
        #Archivage du csv
        dfsave = readCSV(csvpathtemp) 
        dfsave["Archive"] = "Yes"
        dfsave.to_csv(csvpathtemp, index=False)
        
        #Effacer les lignes déjà analysées et celles correspondant à des erreurs
        nonfait = df["Archive"].isna()
        chargmentarrete = df["Erreur"].isna()
        df = df[nonfait & chargmentarrete]
        
        #Pour corriger les types
        for colonne in ["PatientID", "StudyDate", "StudyID"]:
            df[colonne] = df[colonne].astype('float') 
            df[colonne] = df[colonne].astype('int') 
        df["PatientName"] = df["PatientName"].astype('str') 
        df["Surface"] = df["Surface"].astype('float') 
        df["SurfaceGraisseuse"] = df["SurfaceGraisseuse"].astype('float')
        
        #Effacer les erreurs d'apres laprediction du reseau
        dfErreurs = df[(df['PredictionVerification'] == "erreurL3") | (df['PredictionVerification'] == "erreurMuscles")]
        df = df.drop(dfErreurs.index)
        features_cols1 = ["PatientID", "StudyDate", "StudyID", "PatientName","Surface","PatientSex","PatientSize","PredictionVerification"] #,"SurfaceGraisseuse"
        dfErreurs = dfErreurs[features_cols1]

        #Recuperation des colonnes utiles uniquement  
        features_cols2 = ["PatientID", "StudyDate", "StudyID", "PatientName","Surface","PatientSex","PatientSize"] #,"SurfaceGraisseuse"
        df = df[features_cols2]

        nombreODIASP = df.shape[0]
    
    #Chargment des .xls (automatica)
    dfauto, listeExcels = import_all_XLS(dossier)
    if len(listeExcels)>0 :
        nombreAUTOMATICA = dfauto.shape[0]
        dfauto,dfErreursAuto = RecuperationErreurs(dataframe = dfauto, 
                                                   dossierExport = DIR_SORTIE,
                                                   pathdexport = DossierResultasInterm,
                                                   verbose = verbose)
        
    #Fusion des dataframes
    if os.path.isfile(csvpathtemp)==True and len(listeExcels)>0  and nombreODIASP >0:
        #fusion des resultats corrects
        df4 = pandas.concat([df,dfauto], ignore_index=False)
        df4 = df4.groupby(by=["PatientID","StudyDate"],observed=False).agg('mean')
        
        #Affichage des incertitudes sur ces resultats
        df4['Erreurs'] = "Ok"
        df4.loc[df4['Muscle CSA'] < ((1-erreurautorisee)*df4['Surface']), 'Erreurs'] = 'Incertitude car +{}%'.format(erreurautorisee*100) 
        df4.loc[df4['Muscle CSA'] > ((1+erreurautorisee)*df4['Surface']), 'Erreurs'] = 'Incertitude car -{}%'.format(erreurautorisee*100)
        
        #calcul du ratio SAT/VAT
        df4['Ratio_VATsurSAT'] = df4['VAT CSA']/df4['SAT CSA']
        df4['SommeGraisse'] = df4['VAT CSA'] + df4['SAT CSA'] + df4['IMAT CSA']
        
        #On recupere les noms qui ont été perdus par la fonction groupby
        dfnames = df.copy()
        dfnames = dfnames[["PatientID","PatientSex", "StudyDate", "PatientName","PatientSize"]]
        dfnames.drop_duplicates(keep="first",inplace=True) 
        dfnames.set_index(["PatientID","StudyDate"], inplace = True,append = False, drop = True)
        newresult = df4.join(dfnames, how='outer')      
        
        #fusion des resultats erronés                      
        
        AllErrors = pandas.merge(dfErreurs, dfErreursAuto, how='outer', on=["PatientID", "StudyDate"], left_index=True, right_index=True)
        AllErrors.set_index(["PatientID","StudyDate"], inplace = True,append = False, drop = True)
                              
        dfnamesErrors = dfErreurs.copy()
        dfnamesErrors = dfnamesErrors[["PatientID","PatientSex", "StudyDate", "PatientName","PatientSize"]]
        dfnamesErrors.drop_duplicates(keep="first",inplace=True) 
        dfnamesErrors.set_index(["PatientID","StudyDate"], inplace = True,append = False, drop = True)

        AllErrors = AllErrors.merge(dfnamesErrors, how='outer', left_index=True, right_index=True)
        

        #Archivage
        if patharchives!=None:
            destination = "Resultats_EXCEL"
            path = os.path.join(patharchives, destination)

            if os.path.isdir(path)==False:
                try:
                    os.mkdir(path)
                except OSError:
                    print ("Erreur dans la creation du dossier %s" % path)
                else:
                    print ("Creation du dossier %s " % path)

            for f in listeExcels:
                move(os.path.join(dossier,f), path)
            
        if verbose>0: print(nombreODIASP+nombreAUTOMATICA,"segmentations réalisées, ramenées à", newresult.shape[0], "patients uniques,")
    
        #Ecriture du csv
        if os.path.isfile(resultatspath)==True:
            
            ancienresult = pandas.read_csv(resultatspath, delimiter=",")
            ancienresult.set_index(["PatientID","StudyDate"], inplace = True,append = False, drop = True)
            if verbose>0: print("Ajoutées aux", ancienresult.shape[0], "mesures pré-existantes.")
            newresult = pandas.concat([ancienresult,newresult], ignore_index=False)
        newresult.to_csv(resultatspath)
        
        #Ecriture du .xlsx
        if os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(resultatspath)), "Resultats.xlsx"))==True:
            book = openpyxl.load_workbook(os.path.join(os.path.dirname(os.path.realpath(resultatspath)), "Resultats.xlsx"))
            ws = book['Errors']
            data = ws.values
            columns = next(data)[0:]
            AnciennesErreurs = pandas.DataFrame(data, columns=columns)
            AnciennesErreurs.set_index(["PatientID","StudyDate"], inplace = True, append = False, drop = True)
            AllErrors = pandas.concat([AnciennesErreurs,AllErrors], ignore_index=False)
        writer = pandas.ExcelWriter(os.path.join(os.path.dirname(os.path.realpath(resultatspath)), "Resultats.xlsx"))
        newresult.to_excel(writer, sheet_name='main results')
        if AllErrors.shape[0] > 0 :
            AllErrors.to_excel(writer, sheet_name='Errors')
        writer.save()

    else :
        print("Pas de nouveau fichier à charger.")
        if os.path.isfile(resultatspath)==True:
            newresult = pandas.read_csv(resultatspath, delimiter=",")
    
        
    return newresult, AllErrors



#____________________________________________________________________
print("Les fonctions ont été importées")

