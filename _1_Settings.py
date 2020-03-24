# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:00:00 2020

@author: Alexandre NEROT
contact : alexandre@nerot.net

Ceci correspond au fichier de réglages
vous pouvez modifier les lignes marquées par des flèches sans problème
"""
import os

#Indiquer le dossier où trouver les DICOM
GeneralDir   = r"J:\ExportODIASP2" #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#Indiquer le dossier général qui servira de dossier de travail (prévoir plusieurs gigas de libre !)
PATH_PROJECT = r"C:\Users\alexa\IA\ODIASP" #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
"""
Dans ce dossier il faut créer plusieurs sous dossiers pour :
- les images, où l'on sauvegardera les différents volume numpy après lecture des fichiers DICOM
- les masques une fois segmentés dans 3D Slicer
- Les volumes d'image une fois altérés pour rentrer dans le réseau de neurones
- le dataset correspondant à l'ajout un par un de tous les volumes mais également aux data augmentations
tout ceci se fait automatiquement, on peut modifier le nom de ces dossiers mais ce n'est pas nécessaire
"""
PATH_Images   = "Images"     #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PATHduMASK    = "Masks"      #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PATHdEXPORT   = "Scaled"     #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PATH_DATASET  = "Dataset"    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PATH_MODELS   = "models"

"""
Enfin il faut choisir le nom du fichier csv contenant les metadata (il sera créé automatiquement)
"""
CSV_Name    = "metadataTESTGLOBAL.csv" #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#Définit les infos à récupérer dans le DICOM
METADATA = ["PatientName",
            "PatientID",
            "StudyDate",
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
            "Modality"
            
           ]





CLASSTOTEST = ['Abdomen','Cervical','Diaphragme','Membreinf','Pelvien','Thorax']







#Ne pas toucher à ce qui suit
#Ne pas toucher à ce qui suit
#Ne pas toucher à ce qui suit
#Ne pas toucher à ce qui suit
#___________________________________________________________________________________________________________________________
CSV_PATH = os.path.join(os.path.join(PATH_PROJECT,CSV_Name))

