
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
_, _, modelMUSCLE, modelVerification                 = ODIASP.load_modelsODIASP(PATH_MODELS_ODIASP, 
                                                             strategy=strategy, 
                                                             label = False, L3 = False)



finimport2 = time.perf_counter() #TIME
print(finimport2 - finsegm, "secondes pour l'import des reseaux ODIASP")#TIME

ODIASP.PredictMUSCLES(DossierImport     = PATHdEXPORT, #Il s'agit du dossier d'import o'u se situent les images
                      csv_path          = CSV_PATH,
                      Model_segmentation_muscles = modelMUSCLE,
                      Model_Verification = modelVerification,
                      DIR_SORTIE        = PATHduMASK, #Rentrer le nom du dossier dans le dossier gene
                      VERBOSE           = verbose,
                    )

finsegmodiasp = time.perf_counter() #TIME
print(finsegmodiasp - finimport2, "secondes pour la segmentation par ODIASP")#TIME

del modelMUSCLE
del modelVerification
"""
Cette partie du code utilise le reseau et le code écrits par AutoMATiCA :
https://gitlab.com/Michael_Paris/AutoMATiCA
"""
muscle_model, IMAT_model, VAT_model, SAT_model   = AutoMATiCA.load_models(PATH_MODELSAutoMATiCA)


CT_filepaths = AutoMATiCA.generate_CT_filepaths(PATHdEXPORT)

CT_HU, CT_fat_HU, CT_lean_HU, CT_VAT_HU, pred_muscle, pred_IMAT, pred_VAT, pred_SAT, CT_pixel_spacing, CT_image_dimensions, CT_voltage, CT_current, CT_date, CT_slice_thickness, Patient_id, Patient_age, Patient_sex, CT_filepaths, removed_scans,SeriesUID_liste = AutoMATiCA.segmentation_prediction(CT_filepaths, muscle_model, IMAT_model, VAT_model, SAT_model, muscle_range = (-29, 150), VAT_range = (-150, -50), IMAT_SAT_range = (-190, -30), batch_size=int(COMPUTE_CAPACITY))

#combine the 4 predicted segmentation maps into one
combined_map = AutoMATiCA.combine_segmentation_predictions(CT_fat_HU, CT_lean_HU, CT_VAT_HU, pred_VAT, pred_SAT, pred_IMAT, pred_muscle)

#this displays and saves the segmentation images
#AutoMATiCA.display_segmentation(CT_HU, combined_map, Patient_id)

AutoMATiCA.save_segmentation_image(CT_HU, combined_map, Patient_id, save_directory = PATHduMASK)
muscle_CSA, IMAT_CSA, VAT_CSA, SAT_CSA = AutoMATiCA.CSA_analysis(combined_map, CT_pixel_spacing, CT_image_dimensions)
muscle_HU, IMAT_HU, VAT_HU, SAT_HU = AutoMATiCA.HU_analysis(combined_map, CT_HU)

#exports results into an excel file
dfAutomatica = AutoMATiCA.save_results_to_excel(Patient_id, Patient_age, Patient_sex, muscle_CSA, IMAT_CSA, VAT_CSA, SAT_CSA, 
                                                muscle_HU, IMAT_HU, VAT_HU, SAT_HU, CT_pixel_spacing, CT_image_dimensions, 
                                                CT_voltage, CT_current, CT_date, CT_slice_thickness, CT_filepaths, SeriesUID_liste,
                                                DossierDuProjet, removed_scans)


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
df,AllErrors = ODIASP.FusionResultats(DossierDuProjet,
                                      csvpathtemp=CSV_PATH,
                                      DIR_SORTIE = PATHduMASK,
                                      resultatspath=RESULTATS_PATH,
                                      DossierResultasInterm = PATHdEXPORT,
                                      patharchives=PATH_archive,
                                      verbose=1)






finODIASP = time.perf_counter() #TIME
print(finODIASP - finsegm, "secondes pour la totalité")#TIME