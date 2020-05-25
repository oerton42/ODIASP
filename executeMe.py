
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
#strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0)])

if CUPY == True:
    physical_devices = tf.config.list_physical_devices('GPU') 
    try: 
        # Disable first GPU 
        tf.config.set_visible_devices(physical_devices[1:], 'GPU') 
        logical_devices = tf.config.list_logical_devices('GPU') 
        # Logical device was not created for first GPU 
        assert len(logical_devices) == len(physical_devices) - 1 
    except: 
        # Invalid device or cannot modify virtual devices once initialized. 
        pass


"""
    1_ TROUVER L3
"""

start_time = time.perf_counter() #TIME


#Chargement des différents réseaux
modelLABEL, modelL3, modelMUSCLE, _                 = ODIASP.load_modelsODIASP(PATH_MODELS_ODIASP,
                                                                               muscle = SEGM_MUSCLES,
                                                                               verification = False)


ODIASP.All_in_One(dossierDICOM          = DossierImportDICOM,
                  DossierDeTravail      = DossierDuProjet, #C'est ici que se trouvent tous les sous-dossiers des models et les resultats
                  METADATAS             = METADATA,
                  csv_path              = CSV_PATH,
                  MODEL_niveau_de_coupe = modelLABEL,
                  Model_segmentation_L3 = modelL3,
                  Model_segmentation_muscles = modelMUSCLE, #vaut None si le reseau n'a pas été chargé.
                  DIR_SORTIE = PATHdEXPORT,
                  VERBOSE = verbose
                    )



finsegm = time.perf_counter() #TIME
print(finsegm - start_time, "secondes pour la totalité des scanners")#TIME

del modelLABEL
del modelL3

"""
    2_ SEGMENTER LES MUSCLES
"""
