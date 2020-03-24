# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:00:00 2020

@author: Alexandre NEROT
contact : alexandre@nerot.net

Ceci correspond aux fonctions utilisées pour traiter les données dans le cadre du projet ODIASP
"""

COMPUTE_CAPACITY = 12.2


from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
import os
import pandas
from PIL import Image
import scipy
from scipy.ndimage import zoom, center_of_mass
import skimage.io as io
import skimage.transform as trans
import random


def DirVerification (name,DossierProjet=None,verbose = 1):
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
#___________________FONCTIONS POUR NETTOYER UN DOSSIER D'EXPORT DU PACS_____________________
#___________________________________________________________________________________________

fichiers=['AUTORUN.INF','CDVIEWER.EXE','INDEX.HTM','LOG4J.XML','DICOMDIR','RUN.CMD','RUN.COMMAND','RUN.SH','README.TXT']
dossiers=['JRE','HELP','IHE_PDI','PLUGINS','XTR_CONT']

def massacre(path, doss = dossiers, fich = fichiers):
    for i in doss:
        try:
            shutil.rmtree(os.path.join(path,i),ignore_errors=True)
        except:
            pass
    for j in fich:
        try:
            os.remove(os.path.join(path,j))
        except: 
            pass

#___________________________________________________________________________________________
#___________________FONCTIONS POUR IMPORTER LES FICHIERS DICOM______________________________
#___________________________________________________________________________________________



def fast_scandir(dir):
    """
    Pour scanner les sous-dossiers
    Nourrir d'un dossier contenant plusieurs sous dossiers
    sort une liste des PATH pour tous les sous dossiers présents
    Rq : le dossier dir n'est pas inclus dans la liste
    """
    subfolders= [f.path for f in os.scandir(dir) if f.is_dir()]
    for dir in list(subfolders):
        subfolders.extend(fast_scandir(dir))
    return subfolders

def export_dicomDir_to_numpy(rootdir,
                             metadata, #correspond à la liste des Metadata que l'on veut récupérer dans les DICOM 
                             csv_path, #chemin du CSV
                             save=False, #choisir le fait de sauvegarder ou non en numpy array, indiquer le path
                             acceptTAP = True
                             ):
    """
    Nourrir d'un dossier rootdir où se situe des fichiers DICOM, à utiliser avec la fonction fast_scandir
    le CSV sera créé automatiquement si celui-ci n'existe pas encore
    save = False garde les numpy en mémoire.
    remarque : les numpy sont lourds, plus que les DICOM (qui sont compressés), environ 700Mo pour un scan abdo, on peut vite remplir un disque dur en utilisant cete fonction sur un dossier comprenant beaucoup de scanners !
    """
    stopNow = False
    TAP_to_AP = False
    traitement = False
    perdu = 0
    facteur = 1
    inter = {}
    list_files = os.listdir(rootdir)
    if len(list_files) <150:
        print("   Not enough slices")
        stopNow = True
    if len(list_files) >500:
        if acceptTAP == True :
            TAP_to_AP = True
        else :
            stopNow = True
    if stopNow == False:
        echantillon1 = os.path.join(rootdir, list_files[1])
        echantillon2 = os.path.join(rootdir, list_files[100])
        if not os.path.isdir(echantillon1):
            _ds_1 = pydicom.dcmread(echantillon1,force =True, specific_tags =["ImagePositionPatient","SliceThickness","WindowCenter","BodyPartExamined", "FilterType", "SeriesDescription"])
            if (0x18, 0x50) in _ds_1:
                thickness = _ds_1["SliceThickness"].value
                if thickness >2.5: #Limitation si coupe trop épaisses : MIP...etc
                    print("   Thickness is too high.")
                    stopNow = True
            if (0x28, 0x1050) in _ds_1:
                WindowCenter = _ds_1["WindowCenter"].value
                try :
                    if WindowCenter <0:
                        print("   Pulmonary.") #Limitation si fenetre pulmonaire
                        stopNow = True
                except :
                    print("Unexpected error about WindowCenter")
                    print(WindowCenter)
                    stopNow = True #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if (0x18, 0x15) in _ds_1:
                BodyPartExamined = _ds_1["BodyPartExamined"].value
                if "HEAD" in str(BodyPartExamined) : #Limitation si imagerie cerebrale
                    print("   BodyPartExamined : ", BodyPartExamined)
                    stopNow = True
            if (0x18, 0x1160) in _ds_1:
                BodyPartExamined = _ds_1["FilterType"].value
                if "HEAD" in str(BodyPartExamined) : #Limitation si imagerie cerebrale (autre moyen de verification)
                    print("   FilterType : ", BodyPartExamined)
                    stopNow = True
            if (0x8, 0x103E) in _ds_1:
                BodyPartExamined = _ds_1["SeriesDescription"].value
                if "Crane" in str(BodyPartExamined) : #Limitation si imagerie cerebrale (autre moyen de verification)
                    print("   Head.") 
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
            print("   Sagittal plane.")
            stopNow = True
        if position1[1] != position2[1]:
            print("   Coronal plane.")
            stopNow = True
        
    if stopNow == True:
        volume_numpy = np.empty(0)
        return volume_numpy, perdu, facteur #le volume numpy est donc vide si le dossier n'avait pas les informations requises.
        
    if stopNow == False:
        #Maintenant que l'on a arrêté la fonction précocement selon certains criteres, regardons la liste des images
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
        
        #Cette partie de la fonction permet de limiter le nombre de coupes chargées. Cela limite l'import des images DICOM. Cette fonction fait gagner du temps et permet de s'affranchir des inégalités dépaisseurs de coupes.
        nbcoupes = len(liste_fichiers)
        print(nbcoupes, " fichiers trouvés pour ce scanner")
        
        if thickness<1 :
            traitement = True #Indique que le nombre d'image a été traité
            print("Traitement : version coupe fine")
            facteur = 384/nbcoupes
            TAP_to_AP = False
            if nbcoupes >700:
                liste_fichiers= liste_fichiers[-700:]
                perdu = nbcoupes - 700
                facteur = 384/700
                print(perdu, "coupes n'ont pas été chargées")
                
        elif TAP_to_AP == True: #Gestion des scanners trop grands
            traitement = True #Indique que le nombre d'image a été traité
            print("Traitement : version TAP")
            if nbcoupes >500:
                perdu = nbcoupes - 500
                liste_fichiers= liste_fichiers[-500:]
            else :
                perdu = 0
            facteur = 384/500
            print(perdu, "coupes n'ont pas été chargées")
        
        elif nbcoupes >200 and thickness==2.5 : # pour raccourcir les scanners TAP en coupe epaisse
            traitement = True #Indique que le nombre d'image a été traité
            print("Traitement : version coupe epaisse")
            toomuch = nbcoupes - 200
            liste_fichiers= liste_fichiers[-200:]
            perdu = toomuch
            print(perdu, "coupes d'epaisseur de ", thickness, " n'ont pas été chargées")
        
        elif traitement == False : 
            if nbcoupes >384 :
                liste_fichiers= liste_fichiers[-384:]
                perdu = nbcoupes - 384
            else :
                print("Pas de traitement")
    
        nbcoupesfinal = int(len(liste_fichiers)*facteur)
        print(len(liste_fichiers), " ont étés chargées")
        volume_numpy=np.zeros((len(liste_fichiers),x_dim,y_dim))
        slope=float(ds_img1[0x28,0x1053].value)
        intercept=float(ds_img1[0x28,0x1052].value)
        for k in range (0,len(liste_fichiers)):
            dicom_file = pydicom.read_file(liste_fichiers[k])
            img_orig_dcm = (dicom_file.pixel_array)
            img_modif_dcm=(img_orig_dcm*slope) + intercept
            volume_numpy[k,:,:]=img_modif_dcm #ecrit une ligne correspondant à l'image
       
              #Writing on the csv
        titres=""
        if os.path.isfile(csv_path)==False:
            print("Creation du csv")
            with open(csv_path,"wt", encoding="utf-8") as file :
                file.close()
            titres += "Name"
            for ele in metadata:
                titres += ","+ str(ele)
            titres += ",OriginalSlices,Savedslices,Path,Segmented,Scaled,L3Position,Certitude,L3Original,Surface"
        else :
            print("Mise à jour du csv")
        values=str(os.path.basename(rootdir))+".npy"
        for de2 in metadata:
            values = values + "," 
            if de2 in ds_img1:
                if ds_img1[de2].VR == "SQ":
                    values = values + "sequence"
                elif ds_img1[de2].name != "Pixel Data":
                    _ds = str(ds_img1[de2].value)[:64]
                    raw_ds = _ds.replace('\n','__')
                    raw_ds = raw_ds.replace('\r','__')
                    raw_ds = raw_ds.replace('\t',"__")
                    raw_ds = raw_ds.replace(',',"__")
                    values = values + raw_ds
        values += ","+str(nbcoupes)
        values += ","+str(nbcoupesfinal)
        values += ","+str(rootdir)
        with open(csv_path,"a", encoding="utf-8") as file :
                file.write(titres)
                file.write("\n")
                file.write(values)
                file.close() 
        
        volume_numpy = zoom(volume_numpy, (facteur, 1, 1))
        
        print("Le volume est ramené à ", nbcoupesfinal, " coupes")
        #Saving the .npy
        if save != False:
            print("Sauvegarde de "+os.path.basename(rootdir)+".npy ("+str(nbcoupesfinal)+" coupes) dans le dossier "+save)
            np.save(os.path.join(save,os.path.basename(rootdir)),volume_numpy)
    return volume_numpy, perdu, facteur


def TESTINGPRED(prediction,classes,mode="ProportionGlobale", nombredecoupes=1, numerocoupeinitial = 0, verbose=1):
    longueur = len(prediction)
    proportion =0
    if numerocoupeinitial>longueur:
        print("error") #DEBUG
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


    
    
    
def export_dicomDir_to_numpyV2(rootdir,
                               metadata, #correspond à la liste des Metadata que l'on veut récupérer dans les DICOM 
                               csv_path, #chemin du CSV. il sera créé automatiquement
                               save=False, #choisir le fait de sauvegarder ou non en numpy array, indiquer le chemin du dossier 
                               #de sortie
                               classtotest = None,
                               model = None,
                               verbose =2,
                               Compute_capacity = COMPUTE_CAPACITY #Pour adapter automatiquement le calcul réalisé par le reseau en fonction de la capacité de l'ordinateur
                             ):
    """
    Nourrir d'un dossier rootdir où se situe des fichiers DICOM, à utiliser avec la fonction fast_scandir
    le CSV sera créé automatiquement si celui-ci n'existe pas encore
    save = False garde les numpy en mémoire.
    remarque : les numpy sont lourds, plus que les DICOM (qui sont compressés), environ 700Mo pour un scan abdo, on peut vite 
    remplir un disque dur en utilisant cete fonction sur un dossier comprenant beaucoup de scanners !
    
    Ceci est la version V2 : elle utilise un réseau de neurone CNN capable de determiner si une image appartient bien à 
    du Abdominopelvien.
    """
    
    #Verification que le dossier de sortie existe
    if save != False :
        save  = DirVerification (save, verbose = 0)
        
 
    stopNow = False
    perdu = 0
    facteur = 1
    erreur = " "
    inter = {}
    list_files = os.listdir(rootdir)
    if len(list_files) <150:
        erreur = " Pas assez de coupes"
        stopNow = True

    if stopNow == False:
        
        echantillon1 = os.path.join(rootdir, list_files[1])
        echantillon2 = os.path.join(rootdir, list_files[100])
        
        
        if not os.path.isdir(echantillon1):
            _ds_1 = pydicom.dcmread(echantillon1,force =True, specific_tags =["ImagePositionPatient","SliceThickness","WindowCenter","BodyPartExamined", "FilterType", "SeriesDescription","SeriesInstanceUID"])
            
            
            """
            Verifions que cette serie n'a pas deja ete analysee
            """
            if (0x20, 0x000E) in _ds_1:
                NameSerie = str(_ds_1["SeriesInstanceUID"].value)
                NameSerie = NameSerie.replace('.','')
                NameSerie = NameSerie[-30:]
            else :
                NameSerie = "000000000000000000000000000000"
            
            NOMFICHIER = str(os.path.basename(rootdir))+"_"+NameSerie+r".npy"
            if os.path.isfile(csv_path)==True:
                df = readCSV(csv_path,name=None,indexing=True)
                if df.index.str.contains(NameSerie).any() :
                    erreur = " Scanner déjà analysé."
                    stopNow = True
                    
            """
            Nous essayons de déterminer s'il s'agit d'un scanner AP ou TAP à partir uniquement des métadonnées du dicom
            Cela permet si ce n'est pas le cas de réaliser une fonction rapide, qui ne charge aucune image.
            """        
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
                    erreur += " Erreur inconnue sur le WindowCenter."
                    print(WindowCenter)
                    stopNow = True 
                if WindowCenter ==500:
                    erreur += " Fenetrage Os." #Limitation si fenetre os (car trop de grain)
                    stopNow = True
                
            if (0x18, 0x15) in _ds_1:
                BodyPartExamined = _ds_1["BodyPartExamined"].value
                if "HEAD" in str(BodyPartExamined) : #Limitation si imagerie cerebrale
                    erreur += " BodyPartExamined : "+str(BodyPartExamined)
                    stopNow = True
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
        volume_numpy = np.empty(0)
        if verbose>0:print("   Arrêt précoce de niveau 1. Les images n'ont pas été chargées : ",erreur)
        perdu = "Arret"
        facteur = "niveau1"
        return volume_numpy, perdu, facteur, None #le volume numpy est donc vide si le dossier n'avait pas les informations requises.
        
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
        print(len(liste_fichiers), " fichiers trouvés pour ce scanner")
        
        if verbose>0:print("Creation d'un volume echantillon pour labelisation")
        x_dimDIV=x_dim/4
        y_dimDIV=y_dim/4
        ratioECHANTILLONAGE = 5 #Nous allons tester le volume toutes les 5 coupes
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
        volume_pour_label,a,b = normalize(volume_pour_label)
        volume_pour_label     = WL_scaled(WindowCenter,WindowWidth,volume_pour_label,a,b)
            
        if verbose>0:affichage3D(volume_pour_label, 64, axis=2)

        if verbose>0:print("Calcul du volume grâce au reseau de Neurones")
        if verbose >0 : AA=1 #Permet de mettre corriger le verbose si celui_ci était de 2.
        else : AA=0
        AUTO_BATCH = int(Compute_capacity*5.3) #DEBUG test
        prediction = model.predict(volume_pour_label, verbose =AA, batch_size=AUTO_BATCH)
        prediction0_1 = np.zeros_like(prediction, dtype=None, order='K', subok=True, shape=None)
        for i in range (0,np.shape(prediction)[0]):
            prediction0_1[i][np.argmax(prediction[i])] = 1
        
        moyenne = TESTINGPRED(prediction0_1,classtotest,"ProportionGlobale",verbose=0)
        if moyenne[4] <0.1:
            stopNow = True
            erreur = "Le scanner ne possède pas assez de coupes pelviennes"
        
        fin = TESTINGPRED(prediction0_1,classtotest,"ProportionFinale",nombredecoupes=50//ratioECHANTILLONAGE,numerocoupeinitial = 0,verbose=0)
        if (fin[1]+fin[2]+fin[5]) >0.25 : #plus de 25 % finaux sont  cervical, diaphragme ou thorax
            stopNow = True
            erreur = "La fin du volume n'est pas un scanner abdominopelvien"
        else :
            if verbose>1:print("Verifions le debut du volume")
           
    if stopNow == True:
        volume_numpy = np.empty(0)
        if verbose>0:print("   Arrêt précoce de niveau 2. Les images ont été chargées partiellement puis arrêtées : ",erreur)
        perdu = "Arret"
        facteur = "niveau2"
        return volume_numpy, perdu, facteur, None
        
    if stopNow == False:
        total = len(prediction)
        tranchedelecture = 50 #Lecture des 50 premieres coupes
        if verbose>0:print("Analyse des ",tranchedelecture,"premieres coupes échantillonnées à 1/"+str(ratioECHANTILLONAGE)+"...")
        for i in range(0,total,tranchedelecture//ratioECHANTILLONAGE):
            debut = TESTINGPRED(prediction0_1,classtotest,"ProportionInitiale",nombredecoupes=tranchedelecture//ratioECHANTILLONAGE,numerocoupeinitial=i,verbose=0)
            if debut[1] >0.25 : # plus de 25% de crane ou cervical 
                if verbose>1:print("Proportion de crane ou cervical :", debut[1])
                liste_fichiers= liste_fichiers[tranchedelecture:]
                perdu += tranchedelecture
                if verbose>0:print("Supression de ",tranchedelecture," coupes dont ",debut[1]," sont du crane ou cervical.")
        if verbose>0:print("... Pas de crane ou cervical initialement.")
                    
        total = len(prediction) #mise à jour suite à la découpe faite juste au dessus
        tranchedelecture = 50 #Lecture des 50 premieres coupes : on va chercher des coupes thoraciques
        for i in range(0,total,tranchedelecture//ratioECHANTILLONAGE):
            debut = TESTINGPRED(prediction0_1,classtotest,"ProportionInitiale",nombredecoupes=tranchedelecture//ratioECHANTILLONAGE,numerocoupeinitial=i,verbose=0)
            if debut[5]+debut[1] > (debut[0]+debut[4]) : # plus de thorax et cervical que abdopelv #DEBUG +debut[4]+debut[2]
                if verbose>1:print("Proportion de thorax :", debut[5])
                if verbose>1:print("Proportion de abdo :", debut[0]) #DEBUG +pelvien+diaph +debut[2]
                liste_fichiers= liste_fichiers[tranchedelecture:]
                perdu += tranchedelecture
                if verbose>0:print("Supression de ",tranchedelecture," coupes dont la majorité est du thorax.")
        if verbose>0:print("... Pas de thorax majoritaire initialement.")
            
        del volume_pour_label

        
        #Creation du volume representant le scanner dont on garde les coupes
        volume_numpy=np.zeros((len(liste_fichiers),x_dim,y_dim))
        slope=float(ds_img1[0x28,0x1053].value)
        intercept=float(ds_img1[0x28,0x1052].value)
        for k in range (0,len(liste_fichiers)):
            dicom_file = pydicom.read_file(liste_fichiers[k])
            img_orig_dcm = (dicom_file.pixel_array)
            img_modif_dcm=(img_orig_dcm*slope) + intercept
            volume_numpy[k,:,:]=img_modif_dcm #ecrit une ligne correspondant à l'image
        volume_numpy = np.asarray(volume_numpy, dtype=np.float32)
        
        if thickness<1 : #Cette partie de la fonction permet de s'affranchir des inégalités dépaisseurs de coupes.
            facteur = 384/(len(liste_fichiers))
        nbcoupesfinal = int(len(liste_fichiers)*facteur)
        if facteur !=1 :
            if verbose>0:print(len(liste_fichiers)," coupes ont été chargées puis le volume est ramené à ", nbcoupesfinal, " coupes")
            volume_numpy = zoom(volume_numpy, (facteur, 1, 1))
        else : 
            if verbose>0: print(len(liste_fichiers), " coupes ont étés chargées")
    
            
        #Sauvegarde .npy
        if save != False:
            if verbose>0: print("Sauvegarde de "+NOMFICHIER+" ("+str(nbcoupesfinal)+" coupes) dans le dossier "+save)
            np.save(os.path.join(save,NOMFICHIER),volume_numpy)
        
        if verbose>0: 
            if verbose>0: print("...dont voici l'image sagittale centrale")
            affichage3D(volume_numpy, int(x_dim//2), axis=2)    
        
        
        #Ecriture du csv
        titres=""
        if os.path.isfile(csv_path)==False:
            if verbose>0:print("Creation du csv")
            with open(csv_path,"wt", encoding="utf-8") as file :
                file.close()
            titres += "Name"
            for ele in metadata:
                titres += ","+ str(ele)
            titres += ",OriginalSlices,DeletedSlices,Facteur,Path,Segmented,Scaled,L3Position,Certitude,L3Original,Traitement"
        else :
            if verbose>1:print("Mise à jour du fichier csv :", csv_path)

        values=NOMFICHIER
        for de2 in metadata:
            values = values + "," 
            if de2 in ds_img1:
                if ds_img1[de2].VR == "SQ":
                    values = values + "sequence"
                elif ds_img1[de2].name != "Pixel Data":
                    _ds = str(ds_img1[de2].value)[:64]
                    raw_ds = _ds.replace('\n','__')
                    raw_ds = raw_ds.replace('\r','__')
                    raw_ds = raw_ds.replace('\t',"__")
                    raw_ds = raw_ds.replace(',',"__")
                    values = values + raw_ds
        values += ","+str(nbcoupes)
        values += ","+str(perdu)
        values += ","+str(facteur)
        values += ","+str(rootdir)
        with open(csv_path,"a", encoding="utf-8") as file :
                file.write(titres)
                file.write("\n")
                file.write(values)
                file.close() 
        
            
    return volume_numpy, perdu, facteur, NOMFICHIER


#___________________________________________________________________________________________
#___________________FONCTIONS POUR AFFICHAGE DES IMAGES_____________________________________
#___________________________________________________________________________________________

def ApplyWindowLevel (Global_Level,Global_Window,image):
    """
    les scanners sont définis par un réglage d'image avec une largeur de fenetre et un centre
    Cette fonction permet d'afficher une image en 8 bits par pixel 
    utile pour sauvegarder une image png ou jpg mais fait perdre de l'information !!
    """
    li=Global_Level-(Global_Window/2);
    ls=Global_Level+(Global_Window/2);
    image_ret=np.clip(image, li, ls)
    image_ret=image_ret-li
    image_ret=image_ret/(ls-li)
    image_ret=image_ret*255
    #image_ret=image_ret.astype(np.uint8)
    return image_ret

def Norm0_1 (volume_array):
    """
    les scanners ont des voxels dont la valeur est négative, ce qui sera mal interprété pour une image, il faut donc normaliser entre 0 et 1
    """
    a,b,c=volume_array.min(),volume_array.max(),volume_array.mean()
    volume_array_scale=(volume_array-a)/(b-a)
    #print('Min  : ', volume_array_scale.min())
    #print('Max  : ', volume_array_scale.max())
    #print('Mean : ', volume_array_scale.mean())
    return volume_array_scale,a,b,c

def WL_scaled (Global_Level,Global_Window,array,a,b):
    """
    idem que ApplyWindowLevel mais corrigé par les facteurs a et b qui correpsondent au min et max, 
    à utiliser à la place de ApplyWindowLevel si on a utilisé Norm0_1 ou normalize
    """
    li=Global_Level-(Global_Window/2)
    ls=Global_Level+(Global_Window/2)
    li=li/b
    ls=ls/b
    image_ret=np.clip(array, li, ls)
    image_ret=image_ret-li
    image_ret=image_ret/(ls-li)
    #image_ret=image_ret*255
    #image_ret=image_ret.astype(np.uint8)
    return image_ret

#___________________________________________________________________________________________
#___________________FONCTIONS POUR SAVOIR QUOI SEGMENTER et GERER LE CSV____________________
#___________________________________________________________________________________________

def to_segment(pathCSV,number=1,shuffle=False,Colonne="Segmented"):
    """
    Cette fonction permet de trouver des examens dans le fichier cvs qui ne sont pas encore segmentés
    par défaut elle montre le nom d'un examen mais on peut demander autant d'examen que voulus (elle se limitera seule si on demande plus d'examens que disponibles réellement)
    shuffle permet de les obtenir au hasard
    reamrque, en rentrant zero elle permet d'exporter une liste d'examens et le dataset entier. 
    """
    df=pandas.read_csv(pathCSV, delimiter=",")
    if number > df.shape[0]:
        number=df.shape[0]
    Segmented = []
    NotSegmented = []
    for i in range(0,df.shape[0]):
        if pandas.isna(df[Colonne][i]) == True:
            NotSegmented.append(df["Name"][i])
        elif pandas.isna(df[Colonne][i]) != True:
            Segmented.append(df["Name"][i])
    print("There is "+str(len(NotSegmented))+" scans to segment among "+ (str(len(Segmented)+len(NotSegmented))))
    doing=0
    j=0
    if len(NotSegmented) == 0:
        number = 0
    while doing<number:
        if shuffle == True:
            i=np.random.choice(range(0,df.shape[0]),1)[0]
            while pandas.isna(df[Colonne][i]) == False:
                i=np.random.choice(range(0,df.shape[0]),1)[0]
            else:
                print(df.iloc[i])
                i=np.random.choice(range(0,df.shape[0]),1)[0]
                doing +=1
        else:
            if pandas.isna(df[Colonne][i]) == True:
                print(df.iloc[i])
                j+=1
                doing +=1
            else:
                j+=1
    return df, Segmented, NotSegmented

def readCSV(csv_path,name=None,indexing=True):
    """
    Fonction simple pour lire le CSV et le garder en mémoire sous la forme d'un datafile, plus facilement lisible en utilisant pandas
    si on rentre name (un des fichiers numpy disponibles), la fonction affiche cette valeur
    """
    df=pandas.read_csv(csv_path, delimiter=",",dtype=str)
    if indexing == True :
        df.set_index("Name", inplace=True)
    if name:
        print(df.loc[name])
    return df

def DeleteCSVRow(name, csv_path, PATH_PROJECT= r"C:\\Users\\alexa\\OneDrive\\Documents\\ODIASP", raiseErrors=True):
    """
    a nourrir du nom d'un des numpy (string) ou une liste de strings
    Efface dans le csv, dans le dossiers Images et dans le dossier Masks.
    Pour éviter d'effacer une segmentation, par défaut la fonction s'arrête s'il existe un Mask. Il faut modifier raiseErrors pour lever cette sécurité.
    """
    if type(name)==list:
        print(type(name))
        for item in name:
            DeleteCSVRow(name = item, csv_path=csv_path, PATH_PROJECT=PATH_PROJECT, raiseErrors=raiseErrors)
    else :
        if os.path.exists(os.path.join((os.path.join(PATH_PROJECT,"Masks")),name)) and raiseErrors==True:
            raise Exception('{} exists as a mask too.'.format(name))
        df=pandas.read_csv(csv_path, delimiter=",")
        df.set_index("Name", inplace=True)
        try :
            df=df.drop(index=[name])
            df.to_csv(csv_path)
            print("CSV modified")
        except KeyError as error:
            print(error)
        try:
            os.remove(os.path.join(os.path.join(PATH_PROJECT,"Masks"),name))
            print("Mask deleted")
        except FileNotFoundError as error:
            print(error)
        try:
            os.remove(os.path.join(os.path.join(PATH_PROJECT,"Images"),name))
            print("Image deleted")
        except FileNotFoundError as error:
            print(error)
        print("Say goodbye to "+name)
        return df


#___________________________________________________________________________________________
#___________________FONCTIONS POUR VALIDER LA SEGMENTATION__________________________________
#___________________________________________________________________________________________

def Reading_Hardrive (Name, Class=None, dirgeneral= r"C:\\Users\\alexa\\OneDrive\\Documents\\ODIASP"):
    if Class == "Images":
        PATHduNUMPY = os.path.join(os.path.join(dirgeneral,"Images"),Name)
        VOLUME = np.load(PATHduNUMPY)
    elif Class == "Masks":
        PATHduMASK = os.path.join(os.path.join(dirgeneral,"Masks"),Name)
        VOLUME = np.load(PATHduMASK)
    elif Class == None:
        VOLUME = np.load(Name)
    return VOLUME

def Scaling(volume):
    """
    Utilisée dans Reading_and_scale
    suppose un volume en 512,nb_de_coupes,512 : les images scanners sont en 512 par 512 mais le nombre de coupes est variable ++
    permet de segmenter (crop) pour obtenir un volume de 128,384,384
    """
    correction =("ok",1,0)
    volume = volume[192:320,:,96:-32]
    hauteur = np.shape(volume)[1]
    if hauteur<384:
        ratio =384/hauteur
        volume = zoom(volume, (1, ratio, 1))
        correction = ("trop petit",hauteur, ratio)
    if hauteur>384:
        volume = volume[:,-384:,:]
        delta = hauteur-384.
        correction = ("trop grand", hauteur, delta)
    return volume, correction

def normalize (volume_array):
    """
    #Utilisée dans Reading_and_scale
    #Les valeurs des voxels en DICOM sont entre -2000 et +4000, pour limiter les calculs du réseau de neurones il est conseillé de diminuer ces valeurs entre -1 et 1.
    #On sauvegarde les bornes initiales dans les variables a et b.
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
    Utilisée dans Reading_and_scale
    rotation pour passer le volume de axial à sagittal
    """
    volume_array = np.rot90(volume_array,k=sens,axes=(0,1))
    volume_array = np.rot90(volume_array,k=sens,axes=(2,0))
    return volume_array

def sag_to_axial (volume_array, sens=1):
    """
    Utilisée dans Reading_and_scale
    rotation pour passer le volume de axial à sagittal
    """
    volume_array = np.rot90(volume_array,k=sens,axes=(0,2))
    volume_array = np.rot90(volume_array,k=sens,axes=(1,0))
    return volume_array

def Reading_and_scale (name, 
                       NUMPY = "from_hardrive", MASK = "from_hardrive",  #par défaut les volumes NUMPY et MASK sont chargés à partir des dossiers, sinon on peut les fournir en format numpy array.
                       scale = True, #utilise la fonction de scaling, définie par ailleurs
                       downsample = False, #diminue la taille du volume pour gagner en mémoire, rentrer le coefficient comme valeur
                       save = False, #enregistre les resultats 
                       show = False, #affiche des résultats en images
                       csv_path = False, #mettre a jour le csv avec panda
                       mask_path = r"C:\Users\alexa\OneDrive\Documents\ODIASP\Masks",
                       export_path = r"C:\Users\alexa\OneDrive\Documents\ODIASP\Scaled"
                      ):
    """
    Lorsque le mask est fait cette fonction permet de :
    - lire les différents volumes, 
    - vérifier leur concordance
    - modifier leur taille, 
    - mettre à jour le csv 
    - sauvegarder 
    - afficher les résultats pour vérifier que l'enregistrement est bon.
    """
    if NUMPY == "from_hardrive" :
        NUMPY = Reading_Hardrive (name, Class="Images")
    else :
        NUMPY = NUMPY
    if MASK == "from_hardrive" :    
        MASK = Reading_Hardrive (name, Class="Masks")
    else :
        MASK = MASK
    PATHduMASK = os.path.join(mask_path,name)
    PATHdEXPORT = os.path.join(export_path,name)
    
    #inversion du volume du mask, qui a été créé par 3D slicer qui a la mauvaise habitude rendre des volume numpy à l'envers...
    MASK = np.flip(MASK,0)
    MASK = np.flip(MASK,1)
    MASK = np.flip(MASK,2)
    
    if NUMPY.shape!=MASK.shape :
        raise ValueError("Not the same shape")
    print("Initial shape of "+name+":" + str(NUMPY.shape))
    
    #traitement des volumes, cette partie serait à changer en cas de nouveau projet
    MASK = axial_to_sag(MASK) 
    NUMPY = axial_to_sag(NUMPY)
    if scale == True:
        MASK,_ = Scaling(MASK)
        NUMPY,_,_ = normalize(NUMPY)
        NUMPY,_ = Scaling(NUMPY)
    if downsample != False:
        MASK = zoom(MASK, (1, downsample, downsample))
        NUMPY = zoom(NUMPY, (1, downsample, downsample))
        
    if save == True:
        np.save(PATHduMASK, MASK)
        np.save(PATHdEXPORT, NUMPY) #Il s'agit du dossier où seront sauvegardés les volumes des scanners à donner au réseau de neurone
    print("New shape  of "+name+":" + str(NUMPY.shape))
    
    #mettre a jour le csv avec pandas
    if csv_path != False: 
        df=pandas.read_csv(csv_path, delimiter=",")
        df.set_index("Name", inplace=True)
        df.at[name,"Segmented"] = 1
        if scale == True:
            df.at[name,"Scaled"] = 1
        df.to_csv(csv_path)
        
    #affichage des résultats pour vérifier    
    if show == True:
        for k in range (0,(NUMPY.shape)[0],12):
            f = plt.figure()
            image = NUMPY[k,:,:]
            mask = MASK[k,:,:]
            plt.imshow(image,cmap='gray')
            plt.imshow(mask, cmap='magma', alpha=0.7)
            plt.show(block=True)
            
    return NUMPY, MASK

def affichage3D(volume, k, axis=0):
    """
    affiche la coupe k d'un volume, selon son axe axis
    """
    f = plt.figure()
    if axis == 0:
        image1 = volume[k,:,:]
    if axis == 1:
        image1 = volume[:,k,:]
    if axis == 2:
        image1 = volume[:,:,k]
    plt.imshow(image1,cmap='gray')
    plt.show(block=True)
    return
    
def affichage2D(volume):
    """
    affiche une image 2d
    """
    f = plt.figure()
    image1 = volume
    plt.imshow(image1,cmap='gray')
    plt.show(block=True)
    return

def AffichageMulti(volume, frequence, axis=0, FIGSIZE = 40):
    """
    affiche toutes les coupes d'un volume selonl'axe axis, avec une frequence entre les coupes définie
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
#___________________FONCTIONS POUR GESTION DU DATASET_______________________________________
#___________________________________________________________________________________________
    
    
def randomize(a, b):
    """
    Permet de randomiser l'ordre des coupes avant de diviser en train/test/val
    Il faut rentrer les images et les masks pour que leur randomisation soit la même
    """
    permutation = np.random.permutation(a.shape[0])
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b
     
import scipy
def rotation_augmentation(a, b, rotation_max=20, shuffle=True, axis =0, verbose=1):
    """
    prend les volumes a et b et crée deux volumes de mêmes dimensions avec des coupes présentant une rotation
    rotation max définit la rotation possible
    rq : on peut l'utiliser plusieurs fois avec des reglages différents pour ajouter +100% de coupes à chaque fois
    """
    coupes = np.shape(a)[axis]
    a_rotated=np.zeros(np.shape(a))
    b_rotated=np.zeros(np.shape(b))
    min_a,min_b=np.float(a.min()),np.float(b.min())
    for coupe in range(coupes):
        if shuffle==True :
            #creer une valeur au hasard entre -rotation_max et +rotation_max
            rotation_max = abs(rotation_max)
            rotation = np.random.choice(range(-rotation_max,rotation_max),1)[0]
        else :
            rotation = rotation_max
        if axis == 0:
            a_rot = scipy.ndimage.rotate(a[coupe,...], angle=rotation, axes=(0,1), reshape=False, output=None, order=3, mode='constant', cval=min_a, prefilter=True)
            b_rot = scipy.ndimage.rotate(b[coupe,...], angle=rotation, axes=(0,1), reshape=False, output=None, order=3, mode='constant', cval=min_b, prefilter=True)
            a_rotated[coupe,:,:] = a_rot
            b_rotated[coupe,:,:] = b_rot
        if axis == 1:
            a_rot = scipy.ndimage.rotate(a[:,coupe,...], angle=rotation, axes=(0,1), reshape=False, output=None, order=3, mode='constant', cval=min_a, prefilter=True)
            b_rot = scipy.ndimage.rotate(b[:,coupe,...], angle=rotation, axes=(0,1), reshape=False, output=None, order=3, mode='constant', cval=min_b, prefilter=True)
            a_rotated[:,coupe,...] = a_rot
            b_rotated[:,coupe,...] = b_rot
        if verbose == 1:
            print(coupe,"/",coupes)
    return a_rotated, b_rotated
    
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
    Cette fonction sert à créer un fichier dicom à partir des infos contenues dans le CSV et en entrant un volume numpy en entrée.
    On peut l'utiliser seule ou automatiquement dans la fonction PredictL3
    a noter : 
    - en mode name : il faut nourrir l'argument "numpy" avec un string correspondant au nom d'un fichier .npy situé dans le 
    dossier "dossier"
    - en mode dossier : l'arg "numpy" ne sert à rien, la focntion va scanner tout le dossier "dossier" et créer un fichier 
    dicom pour chaque numpy trouvé
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

                
                

def PredictL3 (name,
               model,
               NUMPY = "from_hardrive",  #par défaut le volume est chargés à partir des dossiers, sinon il suffit de nourrir ici un volumenumpy chargé en mémoire
               scale = True, #utilise la fonction de scaling, définie par ailleurs
               downsample = 0.5, #diminue la taille du volume pour gagner en mémoire, rentrer le coefficient comme valeur
               show = True, #affiche des résultats en images
               csv_path = False, #pour mettre a jour le csv avec panda il suffit de donner son adresse ici
               dirgeneral = r"C:\\Users\\alexa\\OneDrive\\Documents\\ODIASP",
               level = 40, #correspond aux réglages de contraste pour l'image de sortie
               window = 400, #correspond aux réglages de contraste pour l'image de sortie
               save=False, #entrer le nom du sous-dossier
               nombredecoupesperdues = 0
                      ):
    """
    Cette fonction prend un volume numpy non traité (correspondant à l'export de ExportfromDicom) et le rentre dans le réseau de neurones.
    On peut appeler ce volume de deux facons : 
    - s'il est sauvegardé en rentrant son nom et en laissant NUMPY= "from_hardrive"
    - s'il est en mémoire en donnant le nom voulu pour name (pas d'importance) et en nourrisant le volume numpy dans l'argument NUMPY.
    level, window permettent de régler le contraste de l'image avant de perdre en qualité (format 8bits)
    la fonction enregistre image (un volume numpy sans perte d'information), et image_wl (une image, prête à être affichée mais avec perte d'information)
    position correspond au numéro de coupe axiale que le réseau estime être le centre de L3
    """
    if NUMPY == "from_hardrive" :
        NUMPY = Reading_Hardrive (name, Class="Images")
    else :
        NUMPY = NUMPY

    #Traitement du volume pour qu'il soit accepté par le reseau de neurones.
    pourimport = axial_to_sag(NUMPY)
    if scale == True:
        pourimport,_,_ = normalize(pourimport)
        pourimport, correction = Scaling(pourimport) #enregistre la 'correction' réalisée dans le scaling
    if downsample != False:
        pourimport = zoom(pourimport, (1, downsample, downsample))
    pourimport = np.asarray(pourimport, dtype=np.float32)
    pourimport = pourimport[:,:,:,np.newaxis]
    prediction = model.predict(pourimport)
    del pourimport
    
    #calcul du centre de gravité du volume donné au réseau pour obtenir le centre de L3 (et correction de sa valeur pour correspondre au volume numpy donné en entrée
    center = scipy.ndimage.center_of_mass(prediction, labels=None, index=None)
    upsample=1/downsample
    position = center[1]*upsample
    if correction[0]=="trop petit": #lit la correction
        position=int(position/correction[2])
    elif correction[0]=="trop grand":
        position=int(position+correction[2])
    print("Axial position :"+str(position), "+", nombredecoupesperdues)
    Standard_deviation = np.std(prediction)
    print(Standard_deviation)
    
    image = NUMPY[position,:,:] #image axiale centrée sur le baricentre
    image_wl=ApplyWindowLevel(level,window,image) #réglages de contraste
    sagittal = NUMPY[:,:,int(center[0])+192] #image sag centrée sur le baricentre
    sagittal_wl=ApplyWindowLevel(level,window,sagittal) #réglages de contraste
    sagittal_wl[position,:] = np.amax(NUMPY)/6 #crée la ligne horizontale sur l'image sag montrant la coupe axiale trouvée 
    if save != False: 
        #Gestion des problèmes de nom de fichier vs nom de dossier
        if str(name)[-4:] == r".npy":
            nameNPY=str(name)[-12:]
            name__=str(name)[-8:-4]
        else:
            name__=str(name)[-8:]
            nameNPY=str(name)[-8:]+r".npy"
        
        #saving the numpy plane
        SAVEPATH = os.path.join(dirgeneral,save)
        np.save(os.path.join(SAVEPATH,nameNPY),image)
        
        #saving the axial image
        arraytopng,_,_,_ = Norm0_1(image_wl)
        arraytopng *=255
        arraytopng= arraytopng.astype(np.uint8)
        im = Image.fromarray(arraytopng)
        im.save(os.path.join(SAVEPATH,name__)+r"_axial.png")
        
        #saving the sagittal image
        sagtopng,_,_,_ = Norm0_1(sagittal_wl)
        sagtopng *=255
        sagtopng= sagtopng.astype(np.uint8)
        im2 = Image.fromarray(sagtopng)
        im2.save(os.path.join(SAVEPATH,name__)+r"_sag.png")
        
        NPY_to_DICOM (numpy=nameNPY, mode="name", csvpath=csv_path, dossier = save, dirgeneral = dirgeneral,numerocoupe = position) 
        
    if show == True:
        _, ax = plt.subplots(1,2,figsize=(25,25))
        ax[0].imshow(image_wl, cmap='gray')
        ax[1].imshow(sagittal_wl, cmap='gray')
        plt.show()

    if csv_path != False: #mettre a jour le csv avec pandas
        df=pandas.read_csv(csv_path, delimiter=",")
        df.set_index("Name", inplace=True)
        df.at[name,"L3Position"] = position
        df.at[name,"Standard_deviation"] = Standard_deviation
        df.to_csv(csv_path)
        
    position +=nombredecoupesperdues
    return image, image_wl, position


#___________________________________________________________________________________________
#___________________FONCTIONS POUR LA VERSION DATA AUGMENT DU RESEAU _______________________
#___________________________________________________________________________________________


def newNorm_and_Scale(VOLUME,downsample = 0.5):
    """
    prend un volume intact et commence à le traiter
    """
    VOLUME = axial_to_sag(VOLUME)
    hauteur = np.shape(VOLUME)[1]
    correction =("ok",1,0)
    if hauteur<384:
        ratio =384/hauteur
        VOLUME = zoom(VOLUME, (1, ratio, 1))
        correction = ("trop petit",hauteur, ratio)
    if hauteur>384:
        VOLUME = VOLUME[:,-384:,:]
        delta = hauteur-384.
        correction = ("trop grand", hauteur, delta)
    VOLUME,a,b = normalize(VOLUME)
    if downsample != 1 :
        VOLUME = zoom(VOLUME, (1, downsample, 1))
    return VOLUME, correction,a,b


def Crop_ram_management(volume, downsample = 0.5):
    """
    Il s'agit d'une fonction utilisée pour diminuer la taille des fichiers
    """
    volume   = volume[96:416,:,96:-32]
    volume   = zoom(volume, (1, 1, downsample))
    return volume


#________________________________________________________________________________________


def PredictL3_v2 (name,
               model,
               NUMPY = "from_hardrive",  #par défaut le volume est chargés à partir des dossiers, sinon il suffit de nourrir 
                  #ici un volumenumpy chargé en mémoire
               downsample =0.5, #diminue la taille du volume pour gagner en mémoire, rentrer le coefficient comme valeur
               csv_path = False, #pour mettre a jour le csv avec panda il suffit de donner son adresse ici
               dirgeneral = r"C:\\Users\\alexa\\OneDrive\\Documents\\ODIASP",
               level = 40, #correspond aux réglages de contraste pour l'image de sortie
               window = 400, #correspond aux réglages de contraste pour l'image de sortie
               savepng=False, #entrer le nom du sous-dossier
                  savedcm=False, #entrer le nom du sous-dossier, peut être le meme que savepng
                  nombredecoupesperdues = 0,
                  facteurAgrandissement = 1,
                  verbose = 2,
                  Compute_capacity = COMPUTE_CAPACITY #Pour adapter automatiquement le calcul réalisé par le reseau en fonction de la capacité de l'ordinateur
                      ):
    """
    Cette fonction prend un volume numpy non traité (correspondant à l'export de ExportfromDicom) et le rentre dans le réseau de neurones.
    On peut appeler ce volume de deux facons : 
    - s'il est sauvegardé en rentrant son nom et en laissant NUMPY= "from_hardrive"
    - s'il est en mémoire en donnant le nom voulu pour name (pas d'importance) et en nourrisant le volume numpy dans l'argument NUMPY.
    level, window permettent de régler le contraste de l'image avant de perdre en qualité (format 8bits)
    la fonction enregistre image (un volume numpy sans perte d'information), et image_wl (une image, prête à être affichée mais avec perte d'information)
    position correspond au numéro de coupe axiale que le réseau estime être le centre de L3
    """
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
    Model_import, correction,a,b = newNorm_and_Scale(NUMPY,downsample=downsample) #enregistre la 'correction' réalisée dans le scaling
    Model_import = Crop_ram_management(Model_import, downsample = downsample) 
    
    #IL FAUT GERER LE CONTRASTE DU VOLUME
    #Model_import = WL_scaled (level,window,Model_import,a,b)
    
    Model_import = Model_import[:,:,:,np.newaxis]
    if verbose>0:a=1 #au cas où verbose = 2 ce qui ne serait pas accepté par la Method predict
    else:a=0
    if verbose>0:print("Recherche de L3 grâce au reseau de Neurones")
    AUTO_BATCH = int(Compute_capacity*1.3) #DEBUG test
    prediction = model.predict(Model_import, verbose =a, batch_size=AUTO_BATCH)
    del Model_import
    
    #calcul du centre de gravité du volume donné au réseau pour obtenir le centre de L3 (et correction de sa valeur pour correspondre au volume numpy donné en entrée
    center = scipy.ndimage.center_of_mass(prediction, labels=None, index=None)
    upsample=1/downsample
    position = center[1]*upsample
    if correction[0]=="trop petit": #lit la correction
        position=int(position/correction[2])
    elif correction[0]=="trop grand":
        position=int(position+correction[2])
    else :
        position = int(position)
    if verbose>0:print("Axial position : "+str(position), "dans ce volume")
    positionreelle = int((position*(1/facteurAgrandissement)) +nombredecoupesperdues)
    if verbose>0 and positionreelle != position : print("Axial position : ",positionreelle, " dans le volume initial")
    Standard_deviation = np.std(prediction)
    Certitude = (Standard_deviation*100)**6
    if verbose>0:print("Estimation de la confiance : ", Certitude)
    
    image = NUMPY[position,:,:] #image axiale centrée sur le baricentre
    image_wl=ApplyWindowLevel(level,window,image) #réglages de contraste
    sagittal = NUMPY[:,:,int(center[0])+96] #image sag centrée sur le baricentre 
    sagittal_wl=ApplyWindowLevel(level,window,sagittal) #réglages de contraste
    sagittal_wl[position,:] = np.amax(NUMPY)/6 #crée la ligne horizontale sur l'image sag montrant la coupe axiale trouvée 
    
    #Gestion des problèmes de nom de fichier vs nom de dossier
    if str(name)[-4:] == r".npy":
        nameNPY=str(name)[-43:]
        name__=str(name)[-43:-4]
    else:
        name__=str(name)[-39:]
        nameNPY=str(name)[-39:]+r".npy"
        
        
    if savepng != False: 
        SAVEPATH = os.path.join(dirgeneral,savepng)
        
        #saving the axial image
        arraytopng,_,_,_ = Norm0_1(image_wl)
        arraytopng *=255
        arraytopng= arraytopng.astype(np.uint8)
        im = Image.fromarray(arraytopng)
        im.save(os.path.join(SAVEPATH,name__)+r"_axial.png")
        
        #saving the sagittal image
        sagtopng,_,_,_ = Norm0_1(sagittal_wl)
        sagtopng *=255
        sagtopng= sagtopng.astype(np.uint8)
        im2 = Image.fromarray(sagtopng)
        im2.save(os.path.join(SAVEPATH,name__)+r"_sag.png")
        
    if savedcm != False: 
        #saving the dicom
        NPY_to_DICOM (numpy=nameNPY, mode="name", csvpath=csv_path, dossier = savedcm, dirgeneral = dirgeneral,numerocoupe = positionreelle) 
        
    if verbose>0:
        _, ax = plt.subplots(1,2,figsize=(25,25))
        ax[0].imshow(image_wl, cmap='gray')
        ax[1].imshow(sagittal_wl, cmap='gray')
        plt.show()
        
    if csv_path != False: #mettre a jour le csv avec pandas
        df=pandas.read_csv(csv_path, delimiter=",")
        df.set_index("Name", inplace=True)
        df.at[nameNPY,"L3Position"] = position #DEBUG pour nameNPY
        df.at[nameNPY,"Standard_deviation"] = Standard_deviation
        df.at[nameNPY,"Certitude"] = Certitude
        df.at[nameNPY,"L3Original"] = positionreelle
        
        df.to_csv(csv_path)

    return image, image_wl, positionreelle





#____________________________________________________________________
print("Les fonctions ont été importées")
