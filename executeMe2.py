
"""

"""

import tensorflow as tf
import FunctionsODIASP     as ODIASP
import FonctionsAutoMATiCA as AutoMATiCA
from Settings import *

import time


PATHduMASK   = ODIASP.DirVerification (PATHduMASK,DossierDuProjet)
PATHdEXPORT  = ODIASP.DirVerification (PATHdEXPORT,DossierDuProjet)
PATH_MODELS_ODIASP     = ODIASP.DirVerification (PATH_MODELS_ODIASP,DossierDuProjet)
PATH_MODELSAutoMATiCA  = ODIASP.DirVerification (PATH_MODELSAutoMATiCA,DossierDuProjet)



"""
Chargement de la strategy
"""

#strategie multiGPU
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


"""
    1_ TROUVER L3
"""
#Deja fait 


finsegm = time.perf_counter() #TIME



"""
    2_ SEGMENTER LES MUSCLES
"""

#Chargement des différents réseaux
_, _, modelMUSCLE                 = ODIASP.load_modelsODIASP(PATH_MODELS_ODIASP, 
                                                             strategy=strategy, 
                                                             label = False, L3 = False)
muscle_model, IMAT_model, VAT_model, SAT_model   = AutoMATiCA.load_models(PATH_MODELSAutoMATiCA)


finimport2 = time.perf_counter() #TIME
print(finimport2 - finsegm, "secondes pour l'import des reseaux ODIASP + automatica")#TIME

ODIASP.PredictMUSCLES(DossierImport     = PATHdEXPORT, #Il s'agit du dossier d'import o'u se situent les images
                      csv_path          = CSV_PATH,
                      Model_segmentation_muscles = modelMUSCLE,
                      DIR_SORTIE        = PATHduMASK, #Rentrer le nom du dossier dans le dossier gene
                      VERBOSE           = verbose,
                    )

finsegmodiasp = time.perf_counter() #TIME
print(finsegmodiasp - finimport2, "secondes pour la segmentation par ODIASP")#TIME

"""
Cette partie du code utilise le reseau et le code écrits par AutoMATiCA :
https://gitlab.com/Michael_Paris/AutoMATiCA
"""


##CT scans can have a .dcm file extension or no file extension
CT_filepaths = AutoMATiCA.generate_CT_filepaths(PATHdEXPORT)


#HU ranges for prediction refine can be narrowed (i.e. muscle from -29 to 150 TO 30 to 150) but cannot be widened (i.e. -29 to 150 TO -50 to 200)
#Batch size indicates # of scans to be analyzed in parallel. For CPU analysis, set to 1 as it does not change speed of analysis. For GPU, increase batch size to increase speed, but will become memory limited at somepoint
CT_HU, CT_fat_HU, CT_lean_HU, CT_VAT_HU, pred_muscle, pred_IMAT, pred_VAT, pred_SAT, CT_pixel_spacing, CT_image_dimensions, CT_voltage, CT_current, CT_date, CT_slice_thickness, Patient_id, Patient_age, Patient_sex, CT_filepaths, removed_scans = AutoMATiCA.segmentation_prediction(CT_filepaths, muscle_model, IMAT_model, VAT_model, SAT_model, muscle_range = (-29, 150), VAT_range = (-150, -50), IMAT_SAT_range = (-190, -30), batch_size=int(COMPUTE_CAPACITY))

#combine the 4 predicted segmentation maps into one
#input is filtered CT scans and predicted segmentation maps
combined_map = AutoMATiCA.combine_segmentation_predictions(CT_fat_HU, CT_lean_HU, CT_VAT_HU, pred_VAT, pred_SAT, pred_IMAT, pred_muscle)

#this displays and saves the segmentation images
#input is raw CT scans, combined maps (overlayed on raw CT scan), patient id, and dir for saving images
#if there are a large number of scans being analzyed at once, displaying the output should be avoided
AutoMATiCA.display_segmentation(CT_HU, combined_map, Patient_id)
AutoMATiCA.save_segmentation_image(CT_HU, combined_map, Patient_id, save_directory = PATHduMASK)

#used for calcualting CSA and average HU for each tissue
#requires combined maps, CT pixel spacing and dimensions for calculations
muscle_CSA, IMAT_CSA, VAT_CSA, SAT_CSA = AutoMATiCA.CSA_analysis(combined_map, CT_pixel_spacing, CT_image_dimensions)
muscle_HU, IMAT_HU, VAT_HU, SAT_HU = AutoMATiCA.HU_analysis(combined_map, CT_HU)

#exports results into an excel file
#will need to update the directory for saving
AutoMATiCA.save_results_to_excel(Patient_id, Patient_age, Patient_sex, muscle_CSA, IMAT_CSA, VAT_CSA, SAT_CSA, muscle_HU, IMAT_HU, VAT_HU, SAT_HU, CT_pixel_spacing, CT_image_dimensions, CT_voltage, CT_current, CT_date, CT_slice_thickness, CT_filepaths, DossierDuProjet, removed_scans)


finsegmAUTO = time.perf_counter() #TIME
print(finsegmAUTO - finsegmodiasp, "secondes pour la segmentation par AutoMATiCA")#TIME


"""
Archivage des images utilisées
"""
PATH_archive  = ODIASP.DirVerification (PATH_archive,DossierDuProjet)
ODIASP.archivage(PATHdEXPORT,PATH_archive,verbose=verbose)




"""
Enregistrement des résultats
"""
df = ODIASP.FusionResultats(DossierDuProjet,
                       csvpathtemp=CSV_PATH,
                       resultatspath=RESULTATS_PATH,
                       patharchives=PATH_archive,
                       verbose=1)


finODIASP = time.perf_counter() #TIME
print(finODIASP - finsegmAUTO, "secondes pour l'archivage et fusion des données")#TIME
print(finODIASP - finsegm, "secondes pour la totalité")#TIME