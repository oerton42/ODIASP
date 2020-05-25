# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:00:00 2020

@author: Alexandre NEROT
contact : alexandre@nerot.net

Ceci correspond au fichier de réglages
"""




#_____________________________________________
#      REGLAGES NECESSAIRES
#_____________________________________________

#Indiquer le dossier où trouver les DICOM
DossierImportDICOM   = r"J:\Exports" #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



#Indiquer le dossier général qui servira de dossier de travail
DossierDuProjet = r"C:\Users\alexa\IA\ODIASP" #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<






















#_____________________________________________
#      REGLAGES OPTIONNELS
#_____________________________________________



#Le nom des sous-dossiers ...
PATHdEXPORT             = "DicomFromODIASP" #Dossier intermédiaire pour enregistrer les fichiers DICOM correspondant aux coupes L3.
PATHduMASK              = "SegmentationMuscles" #Dossier pour sauvegarde des résultats
PATH_MODELS_ODIASP      = "models_ODIASP" #Mettre ici les models d'ODIASP
PATH_MODELSAutoMATiCA   = "models_AutoMATiCA" #Mettre ici les models d'AutoMATiCa
PATH_archive            = "Archives" #C'est le dossier pour l'archivage



# il faut choisir le nom du fichier csv contenant les metadatas (il sera créé automatiquement)
CSV_Name    = "resultatstemporaires.csv" #"Scanners_deja_lus.csv" 
RESULTATS   = "Resultats.csv" 

#Définit les infos à récupérer dans le fichier DICOM
METADATA = ["PatientName",
            "PatientID",
            "StudyDate",
            "StudyID",
            "SeriesInstanceUID",
            "StudyDescription",
            "PatientSex",
            "PatientAge",
            "SliceThickness",
            "Rows",
            "Columns",
            "PixelSpacing",
            "WindowCenter",
            "WindowWidth",
            "PatientSize"
           ]

#Reglage de l'affichage
verbose = 1
# 0 = pas d'affichage, 
# 1 = texte uniquement, 
# 2 = images et texte (mais bloque le script tant que l'image n'est pas fermé si executé dans la console)



#Chosir si on veut segmenter les muscles à la volée ou non :
SEGM_MUSCLES = False



#Réglages de la capacité matérielle de l'ordinateur utilisé
#Correspond à la capacité du GPU, voir le site de nvidia
# https://developer.nvidia.com/cuda-gpus
COMPUTE_CAPACITY = 12.2

#Configuration de la strategy
"""
Il faut configurer la "strategy" >>>>>> dans les fichiers executeMe... .py <<<<<<<
Voir https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy

Par défaut la strategy est configurée pour un ordinateur windows 10 en multigpu.
Le multigpu permet de configurer les calculs des volumes sur une carte et le tensorflow sur la 2eme.
Sans cette distinction cupy et tensofrflow rentrent en conflit.
si vous avez deux cartes et voulez utiliser cupy, gardez Cupy = True, sinon vous pouvez le changer (par tous sauf True).

"""
CUPY = True






#Ne pas toucher à ce qui suit
#Ne pas toucher à ce qui suit
#Ne pas toucher à ce qui suit
#Ne pas toucher à ce qui suit
#___________________________________________________________________________________________________________________________
import os
CSV_PATH = os.path.join(os.path.join(DossierDuProjet,CSV_Name))
RESULTATS_PATH = os.path.join(os.path.join(DossierDuProjet,RESULTATS))