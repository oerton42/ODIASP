"""
Copyright 2019 Michael Paris

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, 
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from skimage.transform import resize
import numpy as np
from tensorflow.python.keras import losses
import pydicom
import os
import datetime

def generate_CT_filepaths(CT_folder_directory):
    
    """
    Recursively pulls the file paths for all CT scans within a folder. Scans within the CT directory should have already 
    been landmarked for the 3rd lumbar vertebrae. 
    
    Parameters
    ----------
    CT_folder_directory : string directing to folder directory 
        CT scans of the 3rd lumbar vertebrae
        
    Returns 
    -------
    CT_filepaths : list
        a list of file paths corresponding to the 3rd lumbar vertebrae scans to be analyzed
    
    Notes
    -----
    Ensure only L3 scans are included in the folder, as all DICOM (.dcm) files will be pulled.
    
    DICOM images without a .dcm file suffix will be added
    
    """

    CT_filepaths = []

    for root, dirs, files in os.walk(CT_folder_directory):
        for file in files:
            base_path, ext= os.path.splitext(os.path.join(root, file))
            
            if ext == '.dcm':
                CT_filepaths.append(os.path.join(root, file))
                
            if not ext:
                os.rename(base_path, base_path + '.dcm') ##add .dcm to DICOM files missing a suffix
                CT_filepaths.append(base_path + '.dcm')
    
    return CT_filepaths

def load_models(model_directory):
    
    """
    Loads previously trained models from a single provided directory folder 
    
    Parameters
    ----------
    model_directory : string directing to model directory
        directory of the folder containing all trained models
        
    Returns
    -------
    muscle_model : tensorflow model
        Prediction of muscle segmentation of L3 scans
    
    IMAT_model : tensorflow model
        Prediction of intermuscular adipose tissue segmentation of L3 scans
    
    VAT_model : tensorflow model
        Prediction of visceral adipose tissue segmentation of L3 scans
    
    SAT_model : tensorflow model
        Prediction of subcutaneous adipose tissue segmentation of L3 scans
    
    Notes
    -----
    Ensure model filenames remain unchanged
    
    """

    ##metrics and loss functions for loading tensorflow models
    def dice_coeff(y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score
    
    def dice_loss(y_true, y_pred):
        loss = 1 - dice_coeff(y_true, y_pred)
        return loss
    
    def bce_dice_loss(y_true, y_pred):
        loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
        return loss
    
    print("\nChargement des différents réseaux AutoMATiCa")

    muscle_model = load_model(os.path.join(model_directory, 'UNET_muscle.h5'), custom_objects= {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
    IMAT_model = load_model(os.path.join(model_directory, 'UNET_IMAT.h5'), custom_objects= {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
    VAT_model = load_model(os.path.join(model_directory, 'UNET_VAT.h5'), custom_objects= {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
    SAT_model = load_model(os.path.join(model_directory, 'UNET_SAT.h5'), custom_objects= {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})

    return muscle_model, IMAT_model, VAT_model, SAT_model

def load_modelsAutoMATiCA_v2(model_directory,strategy):
    
    """
    Loads previously trained models from a single provided directory folder 
    
    Parameters
    ----------
    model_directory : string directing to model directory
        directory of the folder containing all trained models
        
    Returns
    -------
    muscle_model : tensorflow model
        Prediction of muscle segmentation of L3 scans
    
    IMAT_model : tensorflow model
        Prediction of intermuscular adipose tissue segmentation of L3 scans
    
    VAT_model : tensorflow model
        Prediction of visceral adipose tissue segmentation of L3 scans
    
    SAT_model : tensorflow model
        Prediction of subcutaneous adipose tissue segmentation of L3 scans
    
    Notes
    -----
    Ensure model filenames remain unchanged
    
    """

    ##metrics and loss functions for loading tensorflow models
    def dice_coeff(y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score
    
    def dice_loss(y_true, y_pred):
        loss = 1 - dice_coeff(y_true, y_pred)
        return loss
    
    def bce_dice_loss(y_true, y_pred):
        loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
        return loss
    
    print("\nChargement des différents réseaux AutoMATiCa (version modifiée)")
    with strategy.scope():
        muscle_model = load_model(os.path.join(model_directory, 'UNET_muscle'), custom_objects= {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
        IMAT_model = load_model(os.path.join(model_directory, 'UNET_IMAT'), custom_objects= {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
        VAT_model = load_model(os.path.join(model_directory, 'UNET_VAT'), custom_objects= {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
        SAT_model = load_model(os.path.join(model_directory, 'UNET_SAT'), custom_objects= {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})

    return muscle_model, IMAT_model, VAT_model, SAT_model


def segmentation_prediction(CT_filepaths, muscle_model, IMAT_model, VAT_model, SAT_model, muscle_range = (-29, 150), VAT_range = (-150, -50), IMAT_SAT_range = (-190, -30), batch_size=10):
    """
    Applies segementation models to scans in the CT directory
    
    Parameters
    ----------
    CT_filespaths : list
        List of CT filepaths generated from "generate_CT_filespaths"
        
    muscle_model : tensorflow network
        tensorflow network for segmentation of skeletal muscle
    
    IMAT_model : tensorflow network
        tensorflow network for segmentation of IMAT muscle
        
    VAT_model : tensorflow network
        tensorflow network for segmentation of VAT muscle
        
    SAT_model : tensorflow network
        tensorflow network for segmentation of SAT muscle
        
    muscle_range: tupule
        tupule indicating the lower and upper boundry for muscle HU (typically (-29, 150))
        
    VAT_range: tupule
        tupule indicating the lower and upper boundry for VAT HU (typically (-150, -50))
        
    IMAT_SAT_range: tupule
        tupule indicating the lower and upper boundry for SAT and IMAT HU (typically (-190, -30))
        
    Returns
    -------
    CT_HU : numpy array
        Numpy array of HU between -1024 and +1024, normalized between 0 and 1
    
    CT_lean_HU : numpy array
        Numpy array of HU between -29 and +150, normalized between 0 and 1
        
    CT_fat_HU : numpy array
        Numpy array of HU between -190 and -30, normalized between 0 and 1
    
    CT_VAT_HU : numpy array
        Numpy array of HU between -150 and -50, normalized between 0 and 1
    
    pred_muscle : numpy array
        Numpy array probability map of muscle segmentation, between 0 and 1
    
    pred_IMAT : numpy array
        Numpy array probability map of IMAT segmentation, between 0 and 1
    
    pred_VAT : numpy array
        Numpy array probability map of VAT segmentation, between 0 and 1
    
    pred_SAT : numpy array
        Numpy array probability map of SAT segmentation, between 0 and 1
    
    CT_slice_thickness : list
        Thickness of L3 scan (mm)
    
    CT_pixel_spacing : list
        X, Y spacing of each pixel
    
    CT_image_dimensions : list
        X, Y pixel size of each scan

    CT_voltage : list
        CT voltage (KVP)
    
    CT_current : list
        CT tube current 
    
    CT_date : list
        Date CT scan was taken
        
    Patient_id : list
        Patient ID - may be removed during anonomization
    
    Patient_age : list
        Age of patient
    
    Patient_sex : list
        sex of patient(0 = male, 1 = female)
        
    CT_filepaths : list
        updated filepaths to removed non-analyzable RGB CT scans
        
    removed_scans : list
        list of scans that were removed due to RGB format            

    Notes
    -----
    CT scans are resized to 512 X 512 pixels for input into segmentation models
    
    Loads all CT scans at a single time - will be RAM limited, may need to utilize this in batches
    
    """
    #removes RGB dicom scans from the analysis
    removed_scans=[]
    indexs_for_removal = []
    for pop_index, _ in enumerate(CT_filepaths):
        ds = pydicom.read_file(_)
        if len(ds.pixel_array.shape) >2:
            print('\nDICOM file at location:\n %s\nis in an RGB format. It is unable to be processed. This scan will be excluded from analysis' % _)
            removed_scans.append(_)
            indexs_for_removal.append(pop_index)

    for index in sorted(indexs_for_removal, reverse=True):
        CT_filepaths.pop(index)
                
                
                
    CT_dual_muscle = np.empty((len(CT_filepaths), 512, 512, 2)) #used for input into the muscle network [CT_HU, CT_lean_HU]
    CT_dual_fat = np.empty((len(CT_filepaths), 512, 512, 2))    # used for input into the IMAT, VAT, and SAT networks [CT_HU, CT_fat_HU]
    
    CT_HU = np.empty((len(CT_filepaths), 512,512))
    CT_fat_HU = np.empty((len(CT_filepaths), 512,512))
    CT_lean_HU = np.empty((len(CT_filepaths), 512, 512))
    CT_VAT_HU = np.empty((len(CT_filepaths), 512, 512))
    
    CT_pixel_spacing = []
    CT_slice_thickness = []
    CT_image_dimensions = []
    CT_voltage = []
    CT_current = []
    CT_date = []
    Patient_id = []
    Patient_age = []
    Patient_sex = []
      
    print("\nloading CT scans......n=%s" %(len(CT_filepaths)))
    for i, filepath in enumerate(CT_filepaths):
        img = pydicom.read_file(filepath) 
        
        ##pulls CT and patient info for interpretation of body composition data
        try:
            CT_pixel_spacing.append(img.PixelSpacing)
        except:
            CT_pixel_spacing.append('missing from CT scan')
        try:
            CT_slice_thickness.append(img.SliceThickness)
        except:
            CT_slice_thickness.append('missing from CT scan')
        try:
            CT_image_dimensions.append([img.Rows, img.Columns])
        except:
            CT_image_dimensions.append('missing from CT scan')
        try:
            CT_voltage.append(img.KVP)
        except:
            CT_voltage.append('missing from CT scan')
        try:
            CT_current.append(img.XRayTubeCurrent)
        except:
            CT_current.append('missing from CT scan')
        try:
            CT_date.append(img.StudyDate)
        except:
            CT_date.append('missing from CT scan')
        try:
            if not img.PatientID.strip():
                Patient_id.append('no patient ID - index ' + str(i))
            elif img.PatientID.lower() == 'anonymous':
                Patient_id.append('Anonymous ' + str(i))
            else:
                Patient_id.append(img.PatientID)
        except:
            Patient_id.append('no patient ID - index ' + str(i))
        try:
            Patient_age.append(int(''.join(filter(lambda x: x.isdigit(), img.PatientAge))))
        except:
            Patient_age.append('missing from CT scan')
        try:
            if img.PatientSex.lower() == 'm':
                Patient_sex.append('m')
            elif img.PatientSex.lower() == 'f':
                Patient_sex.append('f')
            else:
                Patient_sex.append('missing from CT scan')
        except:
            Patient_sex.append('missing from CT scan')
        
        ##Preprocessing HU and resizes images as required
        if img.pixel_array.shape != [512, 512]:
       
            CT_HU[i] = resize(img.pixel_array, (512,512),  preserve_range=True) #resizes pixel to 512 X 512
            CT_fat_HU[i] = resize(img.pixel_array, (512,512),  preserve_range=True) #resizes pixel to 512 X 512
            CT_VAT_HU[i] = resize(img.pixel_array, (512,512),  preserve_range=True) #resizes pixel to 512 X 512
            CT_lean_HU[i] = resize(img.pixel_array, (512,512),  preserve_range=True) #resizes pixel to 512 X 512
                
                
        else:
            CT_HU[i] = img.pixel_array
            CT_fat_HU[i] = img.pixel_array
            CT_VAT_HU[i] = img.pixel_array
            CT_lean_HU[i] = img.pixel_array
            
            
           
        ##converts pixel data into HU
        CT_HU[i] = CT_HU[i] * int(img.RescaleSlope) 
        CT_HU[i] = CT_HU[i] + int(img.RescaleIntercept)
        
        ##crops HU -1024 to +1024 and normalizes 0 to 1
        CT_HU[i][CT_HU[i]<-1024] = -1024
        CT_HU[i] = CT_HU[i] +1024
        CT_HU[i][CT_HU[i]>2048] = 2048
        CT_HU[i] = CT_HU[i]/2048  
            
        CT_fat_HU[i] = CT_fat_HU[i] * int(img.RescaleSlope)
        CT_fat_HU[i] = CT_fat_HU[i] + int(img.RescaleIntercept)
        CT_fat_HU[i] = CT_fat_HU[i] + 191
        CT_fat_HU[i][CT_fat_HU[i] <0] = 0
        CT_fat_HU[i][CT_fat_HU[i] >161] = 0
        CT_fat_HU[i] = CT_fat_HU[i]/161

        CT_VAT_HU[i] = CT_VAT_HU[i] * int(img.RescaleSlope)
        CT_VAT_HU[i] = CT_VAT_HU[i] + int(img.RescaleIntercept)
        CT_VAT_HU[i] = CT_VAT_HU[i] + 151
        CT_VAT_HU[i][CT_VAT_HU[i] <0] = 0
        CT_VAT_HU[i][CT_VAT_HU[i] >101] = 0
        CT_VAT_HU[i] = CT_VAT_HU[i]/101
 
        CT_lean_HU[i] = CT_lean_HU[i] * int(img.RescaleSlope)
        CT_lean_HU[i] = CT_lean_HU[i] + int(img.RescaleIntercept)
        CT_lean_HU[i] = CT_lean_HU[i] + 30
        CT_lean_HU[i][CT_lean_HU[i] <0] = 0
        CT_lean_HU[i][CT_lean_HU[i] >180] = 0
        CT_lean_HU[i] = CT_lean_HU[i]/180
        
        CT_dual_fat[i] = np.concatenate((CT_HU[i].reshape((512,512,1)), CT_fat_HU[i].reshape((512,512,1))), axis=2) #concatenates CT_HU and CT_fat_HU
        CT_dual_muscle[i] = np.concatenate((CT_HU[i].reshape((512,512,1)), CT_lean_HU[i].reshape((512,512,1))), axis=2) #concatenates CT_HU and CT_lean_HU
        

    print('segmenting muscle')
    pred_muscle = muscle_model.predict(CT_dual_muscle,batch_size,1)
    print('segmenting IMAT')
    pred_IMAT = IMAT_model.predict(CT_dual_fat,batch_size,1)
    print('segmenting VAT')
    pred_VAT = VAT_model.predict(CT_dual_fat,batch_size,1)
    print('segmenting SAT')
    pred_SAT = SAT_model.predict(CT_dual_fat,batch_size,1)

    ##create new HU threshold for prediction refinement based on user input
    ##only able to reduce standard HU ranges, not increase them
    CT_lean_HU = (CT_lean_HU * 180) - 30
    CT_lean_HU = CT_lean_HU + (-muscle_range[0] + 1)
    CT_lean_HU[CT_lean_HU < 0] = 0
    CT_lean_HU[CT_lean_HU > (muscle_range[1] + (-muscle_range[0]+1))] = 0
    CT_lean_HU = CT_lean_HU/(muscle_range[1] + (-muscle_range[0]+1))
    
    CT_VAT_HU = (CT_VAT_HU * 101) - 151
    CT_VAT_HU = CT_VAT_HU + (abs(VAT_range[0])+1)
    CT_VAT_HU[CT_VAT_HU < 0] = 0
    CT_VAT_HU[CT_VAT_HU > (abs(VAT_range[1]) + abs(VAT_range[0])+1)] = 0
    CT_VAT_HU = CT_VAT_HU/(abs(VAT_range[1]) + abs(VAT_range[0])+1)
    
    CT_fat_HU = (CT_fat_HU * 161) - 191
    CT_fat_HU = CT_fat_HU + (abs(IMAT_SAT_range[0])+1)
    CT_fat_HU[CT_fat_HU < 0] = 0
    CT_fat_HU[CT_fat_HU > (abs(IMAT_SAT_range[1]) + abs(IMAT_SAT_range[0])+1)] = 0
    CT_fat_HU = CT_fat_HU/(abs(IMAT_SAT_range[1]) + abs(IMAT_SAT_range[0])+1)

    return CT_HU.reshape(CT_HU.shape[:3]), CT_fat_HU.reshape(CT_HU.shape[:3]), CT_lean_HU.reshape(CT_HU.shape[:3]), CT_VAT_HU.reshape(CT_HU.shape[:3]), pred_muscle.reshape(CT_HU.shape[:3]), pred_IMAT.reshape(CT_HU.shape[:3]), pred_VAT.reshape(CT_HU.shape[:3]), pred_SAT.reshape(CT_HU.shape[:3]), CT_pixel_spacing, CT_image_dimensions, CT_voltage, CT_current, CT_date, CT_slice_thickness, Patient_id, Patient_age, Patient_sex, CT_filepaths, removed_scans


def combine_segmentation_predictions(CT_fat_HU, CT_lean_HU, CT_VAT_HU, pred_VAT, pred_SAT, pred_IMAT, pred_muscle, threshold = 0.1):
    
    """
    Removes incorrectly predicted pixles through thresholding and combining probability maps into a single segmentation map
    
    Parameters
    ----------
    CT_lean_HU : numpy array
        Numpy array of HU between -29 and +150, normalized between 0 and 1
        
    CT_fat_HU : numpy array
        Numpy array of HU between -190 and -30, normalized between 0 and 1
    
    CT_VAT_HU : numpy array
        Numpy array of HU between -150 and -50, normalized between 0 and 1
    
    pred_muscle : numpy array
        Numpy array probability map of muscle segmentation, between 0 and 1
    
    pred_IMAT : numpy array
        Numpy array probability map of IMAT segmentation, between 0 and 1
    
    pred_VAT : numpy array
        Numpy array probability map of VAT segmentation, between 0 and 1
    
    pred_SAT : numpy array
        Numpy array probability map of SAT segmentation, between 0 and 1
    
    Returns
    -------
    combined_map : numpy array
        Numpy array of combined segmentation map (shape = [# of scans, 512, 512]). 0 = background, 1=muscle, 2=IMAT, 3=VAT, 4=SAT
    
    """
    
    print("\nCombining segmentations maps")

    ##creates segment maps to remove incorrectly segmented pixels outside of CT HU ranges
    CT_fat_HU[CT_fat_HU >0] = 1
    CT_VAT_HU[CT_VAT_HU >0] = 1   
    CT_lean_HU[CT_lean_HU >0] = 1    
    
    ##uses threshold of 0.1 as probability pixel segmentation
    pred_muscle[pred_muscle <= threshold] = 0
    pred_IMAT[pred_IMAT <= threshold] = 0
    pred_VAT[pred_VAT <= threshold] = 0
    pred_SAT[pred_SAT <= threshold] = 0
    
    post_y_pred_muscle= CT_lean_HU * pred_muscle
    post_y_pred_IMAT= CT_fat_HU * pred_IMAT
    post_y_pred_VAT= CT_VAT_HU * pred_VAT
    post_y_pred_SAT= CT_fat_HU * pred_SAT
    
    background_map = np.zeros(pred_VAT.shape)
    
    ##stacks segmentation probabilities and sets pixel value according to highest probability (background, muscle, IMAT, VAT, SAT)
    combined_prob_map = np.stack((background_map, post_y_pred_muscle, post_y_pred_IMAT, post_y_pred_VAT, post_y_pred_SAT), axis=-1)
    combined_map = np.argmax(combined_prob_map, axis=-1)
    
    return combined_map


def display_segmentation(CT_HU, combined_map, Patient_id):
    """
    Displays raw HU scan, segmentation map, and overlay of segmentation map on raw HU scan 
    
    Parameters
    ----------
    CT_HU : numpy array
        Numpy array of HU between -1024 and +1024, normalized between 0 and 1
        
    Combined_map : numpy array
        Numpy array of combined segmentation map (shape = [# of scans, 512, 512]). 0 = background, 1=muscle, 2=IMAT, 3=VAT, 4=SAT
        
    Patient_id : list
        Patient ID - may be removed during anonomization
        
    Return
    ------
    overlay : numpy array
    
    RGB_img : numpy array 
    
    CT_display: numpy array
    """
    ##create overlay between raw HU and segmentation map
    RGB_img = np.empty((CT_HU.shape[0], 512, 512, 3), dtype=np.uint8)

    ##creates RGB image from greyscale
    RGB_img[:,:,:,0] = combined_map
    RGB_img[:,:,:,1] = combined_map
    RGB_img[:,:,:,2] = combined_map
    
    ##sets colour map for tissues (muscle - red, IMAT - green, VAT - yellow, SAT - blue)
    for img in range(CT_HU.shape[0]):
        RGB_img[img][(RGB_img[img] == [1, 1, 1]).all(axis=2)] = [255, 0, 0]
        RGB_img[img][(RGB_img[img] == [2, 2, 2]).all(axis=2)] = [0, 255, 0]
        RGB_img[img][(RGB_img[img] == [3, 3, 3]).all(axis=2)] = [255, 255, 0]
        RGB_img[img][(RGB_img[img] == [4, 4, 4]).all(axis=2)] = [64, 224, 208]
    
    CT_display = np.empty((CT_HU.shape[0], 512, 512), dtype=np.uint8)
    overlay = np.empty((CT_HU.shape[0], 512, 512, 4), dtype=np.uint8)
    y_RGBA = np.empty((512, 512, 4), dtype=np.uint8)
    x_RGBA = np.empty((512, 512, 4),  dtype=np.uint8)
    for img in range(CT_HU.shape[0]):
        
        raw_HU = CT_HU[img] * 2048
        raw_HU = raw_HU - 1024
        WW = 600
        WL = 40
        lower_WW = WL-(WW/2)
        upper_WW = WL+(WW/2)
        raw_HU = raw_HU - lower_WW
        raw_HU[raw_HU<=0] = 0
        raw_HU[raw_HU >=(abs(lower_WW) + upper_WW)] = (abs(lower_WW) + upper_WW)
        raw_HU = raw_HU/raw_HU.max()
        raw_HU = raw_HU * 255
        
        CT_display[img] = raw_HU
        
        x_RGBA[:,:,0] = raw_HU
        x_RGBA[:,:,1] = raw_HU
        x_RGBA[:,:,2] = raw_HU
        x_RGBA[:,:,3] = 255
        
        y_RGBA[:,:,0] = RGB_img[img][:,:,0]
        y_RGBA[:,:,1] = RGB_img[img][:,:,1]
        y_RGBA[:,:,2] = RGB_img[img][:,:,2]
        y_RGBA[:,:,3] = 100
        y_RGBA[:,:,3][(RGB_img[img] ==[0, 0, 0]).all(axis=-1)] = 0        
        
        temp_RGBA_dcm = Image.fromarray(x_RGBA, mode='RGBA')
        temp_RGBA_tag = Image.fromarray(y_RGBA, mode='RGBA')
        
        overlay[img] = np.array(Image.alpha_composite(temp_RGBA_dcm, temp_RGBA_tag))

    for i in range(CT_HU.shape[0]):
        fig, axs = plt.subplots(1,3,figsize=(15,5))
        axs = axs.ravel()
        axs[0].imshow(CT_display[i], cmap='Greys_r')
        axs[1].imshow(RGB_img[i])
        axs[2].imshow(overlay[i])
        plt.figtext(x=0.5, y=0.85, s = 'patient ID:' + str(Patient_id[i]))


def save_segmentation_image(CT_HU, combined_map, Patient_id, save_directory):
    """"
    Saves raw HU, blank segmentation, and overlay for examination and display
    
    Parameters
    ----------
    CT_HU : numpy array
        Numpy array of HU between -1024 and +1024, normalized between 0 and 1
        
    Combined_map : numpy array
        Numpy array of combined segmentation map (shape = [# of scans, 512, 512]). 0 = background, 1=muscle, 2=IMAT, 3=VAT, 4=SAT
        
    Patient_id : list
        Patient ID - may be removed during anonomization
        
    save_directory : string of directory
        string directing to the directory for saving 'CT_HU', 'combined_map', and 'overlay' images
    
    Returns
    -------
    N/A - saves images to directory
    """
    
    print('\nsaving raw HU, segmentation maps, and overlay images')

    
   ##create overlay between raw HU and segmentation map
    RGB_img = np.empty((CT_HU.shape[0], 512, 512, 3), dtype=np.uint8)

    ##creates RGB image from greyscale
    RGB_img[:,:,:,0] = combined_map
    RGB_img[:,:,:,1] = combined_map
    RGB_img[:,:,:,2] = combined_map
    
    ##sets colour map for tissues (muscle - red, IMAT - green, VAT - yellow, SAT - blue)
    for img in range(CT_HU.shape[0]):
        RGB_img[img][(RGB_img[img] == [1, 1, 1]).all(axis=2)] = [255, 0, 0]
        RGB_img[img][(RGB_img[img] == [2, 2, 2]).all(axis=2)] = [0, 255, 0]
        RGB_img[img][(RGB_img[img] == [3, 3, 3]).all(axis=2)] = [255, 255, 0]
        RGB_img[img][(RGB_img[img] == [4, 4, 4]).all(axis=2)] = [64, 224, 208]
    
        
    CT_display = np.empty((CT_HU.shape[0], 512, 512), dtype=np.uint8)
    overlay = np.empty((CT_HU.shape[0], 512, 512, 4), dtype=np.uint8)
    y_RGBA = np.empty((512, 512, 4), dtype=np.uint8)
    x_RGBA = np.empty((512, 512, 4),  dtype=np.uint8)
    for img in range(CT_HU.shape[0]):
        
        raw_HU = CT_HU[img] * 2048
        raw_HU = raw_HU - 1024
        WW = 600
        WL = 40
        lower_WW = WL-(WW/2)
        upper_WW = WL+(WW/2)
        raw_HU = raw_HU - lower_WW
        raw_HU[raw_HU<=0] = 0
        raw_HU[raw_HU >=(abs(lower_WW) + upper_WW)] = (abs(lower_WW) + upper_WW)
        raw_HU = raw_HU/raw_HU.max()
        raw_HU = raw_HU * 255
        
        CT_display[img] = raw_HU
        
        x_RGBA[:,:,0] = raw_HU
        x_RGBA[:,:,1] = raw_HU
        x_RGBA[:,:,2] = raw_HU
        x_RGBA[:,:,3] = 255
        
        y_RGBA[:,:,0] = RGB_img[img][:,:,0]
        y_RGBA[:,:,1] = RGB_img[img][:,:,1]
        y_RGBA[:,:,2] = RGB_img[img][:,:,2]
        y_RGBA[:,:,3] = 100
        y_RGBA[:,:,3][(RGB_img[img] ==[0, 0, 0]).all(axis=-1)] = 0        
        
        temp_RGBA_dcm = Image.fromarray(x_RGBA, mode='RGBA')
        temp_RGBA_tag = Image.fromarray(y_RGBA, mode='RGBA')
        
        overlay[img] = np.array(Image.alpha_composite(temp_RGBA_dcm, temp_RGBA_tag))
    
    ##creates directory for saving images if not already created

    try:
        os.mkdir(os.path.join(save_directory, 'raw CT scan'))
        os.mkdir(os.path.join(save_directory, 'segmentation map'))
        os.mkdir(os.path.join(save_directory, 'overlay'))
    except:
        pass
        
    ##saves each image to specific directory
    for scan in range(CT_HU.shape[0]):
        if os.path.exists(os.path.join(save_directory, 'raw CT scan', Patient_id[scan] + '.png')):
            filename_CT_HU = os.path.join(save_directory, 'raw CT scan', Patient_id[scan] + str(scan) + '.png') 
            plt.imsave(filename_CT_HU, CT_display[scan], cmap='Greys_r')
        else:
            filename_CT_HU = os.path.join(save_directory, 'raw CT scan', Patient_id[scan] + '.png') 
            plt.imsave(filename_CT_HU, CT_display[scan], cmap='Greys_r')
        
        if os.path.exists(os.path.join(save_directory, 'segmentation map', Patient_id[scan] + '.png')):
            filename_combined = os.path.join(save_directory, 'segmentation map', Patient_id[scan]+ str(scan) + '.png') 
            plt.imsave(filename_combined, RGB_img[scan])            
        
        else:
            filename_combined = os.path.join(save_directory, 'segmentation map', Patient_id[scan] + '.png') 
            plt.imsave(filename_combined, RGB_img[scan])
        
        if os.path.exists(os.path.join(save_directory, 'overlay', Patient_id[scan] + '.png')):
            filename_overlay = os.path.join(save_directory, 'overlay', Patient_id[scan] + str(scan) + '.png')
            plt.imsave(filename_overlay, overlay[scan])
        
        else:   
            filename_overlay = os.path.join(save_directory, 'overlay', Patient_id[scan] + '.png')
            plt.imsave(filename_overlay, overlay[scan])    

def CSA_analysis(combined_map, CT_pixel_spacing, CT_image_dimensions):
    """
    Quantifies CSA (muscle, IMAT, VAT, SAT) of segmented CT scans
    
    Parameters
    ----------
    combined_map : numpy array
        Numpy array of combined segmentation map (shape = [# of scans, 512, 512]), 0 = background, 1=muscle, 2=IMAT, 3=VAT, 4=SAT
    
    CT_pixel_spacing : list
        List containing the X,Y pixel spacing for each scan
        
    CT_image_dimensions : list
        list containing pixel sizes of the total image
        
    Returns
    -------
    muscle_CSA : list
        list of muscle CSA for each scan (mm)
    
    IMAT_CSA : list
        list of IMAT CSA for each scan (mm)
        
    VAT_CSA : list
        list of VAT CSA for each scan (mm)
        
    SAT_CSA : list
        list of SAT CSA for each scan (mm)
    
    Notes
    -----
    CT scans of 512 X 512, 1024 X 1024, and 2048 X 2048 have been implemented
    """
    muscle_CSA = []
    IMAT_CSA = []
    VAT_CSA = []
    SAT_CSA = []
    
    print('\ncalculating CSA')
    
    for x in range(combined_map.shape[0]):
        if CT_image_dimensions[x]== [512,512]:
            ##counts all non-zero pixels for segmentation map == 1 (for muscle) and pixel count by pixel spacing
            muscle_CSA.append((float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==1))/100)
            IMAT_CSA.append((float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==2))/100)
            VAT_CSA.append((float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==3))/100)
            SAT_CSA.append((float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==4))/100)

        elif CT_image_dimensions[x]== [1024,1024]:
            muscle_CSA.append((4*(float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==1)))/100)
            IMAT_CSA.append((4*(float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==2))/100))
            VAT_CSA.append((4*(float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==3))/100))
            SAT_CSA.append((4*(float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==4))/100))
           
        elif CT_image_dimensions[x]== [2048, 2048]:
            muscle_CSA.append((16*(float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==1)))/100)
            IMAT_CSA.append((16*(float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==2))/100))
            VAT_CSA.append((16*(float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==3))/100))
            SAT_CSA.append((16*(float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(combined_map[x]==4))/100))
       
        else:
            muscle_CSA.append((float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(resize(combined_map[x]==1, CT_image_dimensions[x],  order=0, preserve_range=True))/100))
            IMAT_CSA.append((float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(resize(combined_map[x]==2, CT_image_dimensions[x],  order=0, preserve_range=True))/100))
            VAT_CSA.append((float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(resize(combined_map[x]==3, CT_image_dimensions[x],  order=0, preserve_range=True))/100))
            SAT_CSA.append((float(CT_pixel_spacing[x][0]**2) * np.count_nonzero(resize(combined_map[x]==4, CT_image_dimensions[x],  order=0, preserve_range=True))/100))
             
    return muscle_CSA, IMAT_CSA, VAT_CSA, SAT_CSA

def HU_analysis(combined_map, CT_HU):
    """
    Quantifies average HU ('quality') of muscle, IMAT, VAT, SAT of segmented CT scans
    
    Parameters
    ----------
    combined_map : numpy array
        Numpy array of combined segmentation map (shape = [# of scans, 512, 512]), 0 = background, 1=muscle, 2=IMAT, 3=VAT, 4=SAT
    
    CT_HU : numpy array
        Numpy array of HU between -1024 and +1024, normalized between 0 and 1
        
    Returns
    -------
    muscle_HU : list
        list of muscle HU for each scan (HU)
    
    IMAT_HU : list
        list of IMAT HU for each scan (HU)
        
    VAT_HU : list
        list of VAT HU for each scan (HU)
        
    SAT_HU : list
        list of SAT HU for each scan (HU)
    
    """
    muscle_HU = []
    IMAT_HU = []
    VAT_HU = []
    SAT_HU = []
    
    print('\ncalculating HU average')
    
    ##temporary storage of tissue specific map
    muscle_map = np.empty(combined_map.shape)
    IMAT_map = np.empty(combined_map.shape)
    VAT_map = np.empty(combined_map.shape)
    SAT_map = np.empty(combined_map.shape)
    
    for scan in range(combined_map.shape[0]):
        muscle_map[scan] = combined_map[scan]
        IMAT_map[scan] = combined_map[scan]
        VAT_map[scan] = combined_map[scan]
        SAT_map[scan] = combined_map[scan]

    ##sets non tissue specific pixels to 0 and tissue specific pixels to 1
    muscle_map[muscle_map != 1] = 0
    IMAT_map[IMAT_map != 2] = 0
    IMAT_map = IMAT_map/2
    VAT_map[VAT_map !=3] = 0
    VAT_map = VAT_map/3
    SAT_map[SAT_map!=4] = 0
    SAT_map = SAT_map/4
    
    ##sets HU back to raw values
    raw_HU = CT_HU * 2048
    raw_HU = raw_HU - 1024
    
    ##segments specific tissue of interest and calculates average HU
    for scan in range(combined_map.shape[0]):
        muscle_mask = muscle_map[scan] * raw_HU[scan]
        muscle_HU.append(muscle_mask[muscle_mask != 0].mean())

        IMAT_mask = IMAT_map[scan] * raw_HU[scan]
        IMAT_HU.append(IMAT_mask[IMAT_mask != 0].mean())
        
        VAT_mask = VAT_map[scan] * raw_HU[scan]
        VAT_HU.append(VAT_mask[VAT_mask != 0].mean())
        
        SAT_mask = SAT_map[scan] * raw_HU[scan]
        SAT_HU.append(SAT_mask[SAT_mask != 0].mean())
        
    return muscle_HU, IMAT_HU, VAT_HU, SAT_HU
    
def save_results_to_excel(Patient_id, Patient_age, Patient_sex, muscle_CSA, IMAT_CSA, VAT_CSA, SAT_CSA, muscle_HU, IMAT_HU, VAT_HU, SAT_HU, CT_pixel_spacing, CT_image_dimensions, CT_voltage, CT_current, CT_date, CT_slice_thickness, CT_filepaths, results_directory, removed_scans):
    """
    Saves results to an excel file
    
    Parameters
    ----------
    Patient_id : list
        Patient ID - may be removed during anonomization
    
    Patient_age : list
        Age of patient
    
    Patient_sex : list
        sex of patient(0 = male, 1 = female)
        
    muscle_CSA : list
        list of muscle CSA for each scan (mm)
    
    IMAT_CSA : list
        list of IMAT CSA for each scan (mm)
        
    VAT_CSA : list
        list of VAT CSA for each scan (mm)
        
    SAT_CSA : list
        list of SAT CSA for each scan (mm)
        
    muscle_HU : list
        list of muscle HU for each scan (HU)
    
    IMAT_HU : list
        list of IMAT HU for each scan (HU)
        
    VAT_HU : list
        list of VAT HU for each scan (HU)
        
    SAT_HU : list
        list of SAT HU for each scan (HU)
        
    CT_slice_thickness : list
        Thickness of L3 scan (mm)
    
    CT_pixel_spacing : list
        X, Y spacing of each pixel
    
    CT_image_dimensions : list
        X, Y pixel size of each scan

    CT_voltage : list
        CT voltage (KVP)
    
    CT_current : list
        CT tube current 
    
    CT_date : list
        Date CT scan was taken
        
    CT_filepaths : list
        list of file paths of L3 scans
        
    removed_scans : list
        list of scans that were removed due to RGB format    

    Returns
    -------
    N/A - saves excel spreadsheet of results
    """
    
    results_dataframe = pd.DataFrame({'Patient ID': Patient_id,
                                      'Age': Patient_age,
                                      'Sex': Patient_sex,
                                      'Muscle CSA': muscle_CSA,
                                      'IMAT CSA': IMAT_CSA,
                                      'VAT CSA': VAT_CSA,
                                      'SAT CSA': SAT_CSA,
                                      'Muscle HU': muscle_HU,
                                      'IMAT HU': IMAT_HU,
                                      'VAT HU': VAT_HU,
                                      'SAT HU': SAT_HU,
                                      'CT pixel spacing': CT_pixel_spacing,
                                      'CT image dimensions': CT_image_dimensions,
                                      'CT voltage': CT_voltage,
                                      'CT current': CT_current,
                                      'CT date': CT_date,
                                      'CT slice thickness': CT_slice_thickness,
                                      'Scan folder': CT_filepaths})
    

    date = datetime.datetime.now()
    
    writer = pd.ExcelWriter(os.path.join(results_directory, "Results - " + date.ctime().replace(':', '-') + '.xlsx'))
    results_dataframe.to_excel(writer, sheet_name='main results')
        
    if len(removed_scans) > 0:
        removed_dataframe = pd.DataFrame({'scans that were not analyzed': removed_scans})
        removed_dataframe.to_excel(writer, sheet_name='scans removed during analysis')
    else:
        pass
    
    
    writer.save()
    
    