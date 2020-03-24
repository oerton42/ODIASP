#!/usr/bin/env python
# coding: utf-8


#cette fonction, spécifique à jupyter notebook permet d'importer les fonctions contenues dans un fichier séparé
import _1_Settings-SEMG
import ODIASP_functions




# <h1>CREATION DES FICHIERS</h1>


#separation des dicom et des tags
DOSSIER = r"J:\ODIASP Muscle\coupes_L3_anno"
LISTE=os.listdir(DOSSIER)

import shutil
for item in LISTE:
    if str(item)[-4:]==".dcm":
        shutil.move(os.path.join(DOSSIER,item), os.path.join(PATH_Images,item))
                

#modification des tags en .txt
DOSSIER = r"J:\ODIASP Muscle\coupes_L3_anno"
LISTE=os.listdir(DOSSIER)

for item in LISTE:
    nom = str(item)
    nom += ".txt"
    os.rename(os.path.join(DOSSIER,item), os.path.join(DOSSIER,nom))

#Modification des .txt pour supprimer les infos inutiles et transformer le contenu (binary) en numpy array :

DOSSIER = r"J:\ODIASP Muscle\coupes_L3_anno"
LISTE=os.listdir(DOSSIER)

import struct

for item in LISTE:
    with open(os.path.join(DOSSIER,item), 'r') as file:
        name = str(item)[:-8]
        name += ".npy"
        contenu = file.read()
        contenu = contenu[288:]
        contenu2 = contenu.encode('ascii')
        contenu2 = struct.unpack("B"*262144,contenu2)
        contenu = np.array(contenu2)
        contenu.shape = (512, 512)
        
        SAVEPATH = os.path.join(PATHduMASK,name)
        np.save(SAVEPATH,contenu)
        contenu *=255
        im2 = Image.fromarray(contenu)
        affichage2D(im2)


#Conversion des images dicom en numpy

LISTE=os.listdir(PATH_Images)

for item in LISTE:
    
    dicom_file = pydicom.dcmread(os.path.join(PATH_Images,item), force=True)
    dicom_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    img_orig_dcm = (dicom_file.pixel_array)
    slope=float(dicom_file[0x28,0x1053].value)
    intercept=float(dicom_file[0x28,0x1052].value)
    img_modif_dcm=(img_orig_dcm*slope) + intercept
    WindowCenter = 40
    WindowWidth = 400

    img_modif_dcm=ApplyWindowLevel(WindowCenter,WindowWidth,img_modif_dcm) #réglages de contraste
    name = str(item)
    name += ".npy"
        
    SAVEPATH = os.path.join(PATH_Images,name)
    np.save(SAVEPATH,img_modif_dcm)
    print(SAVEPATH)
    affichage2D(img_modif_dcm)



#Modification des numpy pour les rendre lisibles par le reseau
LISTE=os.listdir(PATH_Images)

for item in LISTE:
    if str(item)[-4:] != ".dcm" :
        print(os.path.join(PATH_Images,item))
        volume = Reading_Hardrive (item, Class="Images", dirgeneral= PATH_PROJECT)
        #affichage2D(volume)
        volume = volume/volume.max()
        affichage2D(volume)
        volume = volume[np.newaxis,:,:,np.newaxis]
        volume = np.asarray(volume, dtype=np.float32)
        SAVEPATH = os.path.join(r"J:\IA\ODIASP2\Dataset\Images",item)
        np.save(SAVEPATH,volume)



#Modification des numpy MASK pour les rendre lisibles par le reseau
LISTE=os.listdir(PATHduMASK)

for item in LISTE:
    if str(item)[-4:] != ".dcm" :
        print(os.path.join(PATHduMASK,item))
        volume = Reading_Hardrive (item, Class="Masks", dirgeneral= PATH_PROJECT)
        volume = volume[np.newaxis,:,:,np.newaxis]
        volume = np.asarray(volume, dtype=np.float32)
        SAVEPATH = os.path.join(r"J:\IA\ODIASP2\Dataset\Masks",item)
        np.save(SAVEPATH,volume)


#Pour transformer directement les DICOM en png
LISTE=os.listdir(PATH_Images)

for item in LISTE:
    if str(item)[-4:] == ".dcm" :
        dicom_file = pydicom.dcmread(os.path.join(PATH_Images,item), force=True)
        dicom_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        img_orig_dcm = (dicom_file.pixel_array)
        slope=float(dicom_file[0x28,0x1053].value)
        intercept=float(dicom_file[0x28,0x1052].value)
        img_modif_dcm=(img_orig_dcm*slope) + intercept
        WindowCenter = 40
        WindowWidth = 400

        img_modif_dcm=ApplyWindowLevel(WindowCenter,WindowWidth,img_modif_dcm) #réglages de contraste
        name = str(item)
        name += ".npy.png"
        im2 = Image.fromarray(img_modif_dcm)
        im2 = im2.convert("L")

        SAVEPATH = os.path.join(r"J:\IA\ODIASP2\Dataset\Imagespng",name)
        im2.save(SAVEPATH)

#Convertir les numpy masks en png
    
LISTE=os.listdir(PATHduMASK)
    
for item in LISTE:
    if str(item)[-4:] != ".dcm" :
        print(os.path.join(PATH_Images,item))
        volume = Reading_Hardrive (item, Class="Masks", dirgeneral= PATH_PROJECT)
        #affichage2D(volume)
        #volume = volume/volume.max()
        
        volume *=255
        #affichage2D(volume)
        #volume= volume.astype(np.uint8)
        
        
        name = str(item)
        name += ".png"
        
        im2 = Image.fromarray(volume)
        im2 = im2.convert("L")

        SAVEPATH = os.path.join(r"J:\IA\ODIASP2\Dataset\Maskspng",name)
        im2.save(SAVEPATH)
        
        
        
        
        
        
        


# <h1>CREATION DU DATASET</h1>


from __future__ import absolute_import, division, print_function, unicode_literals
import scipy
from scipy.ndimage import zoom, center_of_mass
import tensorflow as tf
import skimage.io as io
import skimage.transform as trans
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras.backend as K


import pydicom as dicom
from PIL import Image



DSMASKS= r"J:\IA\ODIASP2\Dataset\Maskspng"
DSIMAGES= r"J:\IA\ODIASP2\Dataset\Imagespng"

BATCH_SIZE = 2
EPOCHS = 10
TARGETSIZE = (512,512)



import imageio

NAME = r"158373.dcm.npy.png"
MASK = os.path.join(r"J:\IA\ODIASP2\Dataset\Maskspng\Dossier",NAME)
IMAGE = os.path.join(r"J:\IA\ODIASP2\Dataset\Imagespng\Dossier",NAME)

#lecture image -> numpy
masque=imageio.imread(MASK)
image=imageio.imread(IMAGE)

print(type(masque), np.shape(masque))

masque = masque[np.newaxis,:,:,np.newaxis]
image = image[np.newaxis,:,:,np.newaxis]




data_gen_args = dict(rotation_range=30.,
                     horizontal_flip = True,
                     #vertical_flip = True,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.3,
                     fill_mode = 'nearest',
                     validation_split = 0.1, #0.15
                     rescale = 1/255.
                     #samplewise_center=True,
                    )

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 20

image_datagen.fit(image, augment=True, seed=seed)
mask_datagen.fit(masque, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(DSIMAGES,
                                                    class_mode=None,
                                                    color_mode = "grayscale",
                                                    subset = "training",
                                                    target_size=TARGETSIZE,
                                                    batch_size=BATCH_SIZE,
                                                    seed=seed)

mask_generator = mask_datagen.flow_from_directory(DSMASKS,
                                                  class_mode=None,
                                                  color_mode = "grayscale",
                                                  subset = "training",
                                                  target_size=TARGETSIZE,
                                                  batch_size=BATCH_SIZE,
                                                  seed=seed)

image_generator_val = image_datagen.flow_from_directory(DSIMAGES,
                                                    class_mode=None,
                                                    color_mode = "grayscale",
                                                    subset = "validation",
                                                    target_size=TARGETSIZE,
                                                    batch_size=BATCH_SIZE,
                                                    seed=seed)

mask_generator_val = mask_datagen.flow_from_directory(DSMASKS,
                                                  class_mode=None,
                                                  color_mode = "grayscale",
                                                  subset = "validation",
                                                  target_size=TARGETSIZE,
                                                  batch_size=BATCH_SIZE,
                                                  seed=seed)

train_generator = zip(image_generator, mask_generator)
val_generator = zip(image_generator_val, mask_generator_val)







# <h1>CREATION DU RESEAU et ENTRAINEMENT</h1>



#strategie multiGPU
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],
                                          cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


"""
Un essai pour avoir une loss avec des poids mais je n'ai pas réussi à l'utiliser. Voir section suivante
"""
# weight: weighted tensor(same shape with mask image)
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    
    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) *     (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) +     weighted_dice_loss(y_true, y_pred, weight)
    return loss


weight = tf.constant(0.5)
#cross_entropy_weighted_loss = tf.nn.weighted_cross_entropy_with_logits(x_function, targets, weight)
#cross_entropy_weighted_out = sess.run(cross_entropy_weighted_loss)

pos_weight = tf.constant(0.5)




"""
loss avec des poids, celle que j'utilise
"""
#https://stackoverflow.com/questions/60253082/weighting-samples-in-multiclass-image-segmentation-using-keras


def balanced_cross_entropy(beta):
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
        return tf.math.reduce_mean(loss * (1 - beta))

    return loss




def unet(
    pretrained_weights = None,
    input_size = (512,512,1) #(192,192,1)
):
    inputs = Input(input_size)
    initial = 96
    initx2 = initial * 2
    initx4 = initx2 * 2
    initx8 = initx4 * 2
    initx16 = initx8 * 2
    
    conv1 = Conv2D(initial, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(initial, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(initx2, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(initx2, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(initx4, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(initx4, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(initx8, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(initx8, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(initx16, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(initx16, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(initx8, 2, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(initx8, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(initx8, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(initx4, 2, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(initx4, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(initx4, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(initx2, 2, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(initx2, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(initx2, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(initial, 2, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(initial, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(initial, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'selu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer=Adam(lr = 1e-4), loss = balanced_cross_entropy(0.2), metrics = ['accuracy'])
    #model.compile(optimizer = SGD(lr=0.0001, decay=1e-5, momentum=0.9, nesterov=True), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #Adam(lr = 1e-4)

    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model




with strategy.scope():
    modelSegMuscles = unet(pretrained_weights = None)



from tensorflow.keras.callbacks import LearningRateScheduler

class LearningRateReducerCb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * 0.9
        print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
        self.model.optimizer.lr.assign(new_lr)

callbacks=[LearningRateReducerCb()]



hist= modelSegMuscles.fit(train_generator,
                steps_per_epoch=647//BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=val_generator,
                #callbacks=callbacks,
                validation_steps=71//BATCH_SIZE
                               )


# In[15]:



# Plot training & validation accuracy values
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[25]:


tf.saved_model.save(modelSegMuscles,os.path.join(PATH_MODELS,"ModelSEG_Muscles_V3_40epochs"))
print(PATH_MODELS)


# In[75]:


modelSegMuscles.summary()


# In[16]:


import imageio

LISTE = os.listdir(r"J:\IA\ODIASP\L3_augment2")
LISTE_PNG = []
LISTE_DCM = []

for k in range (0,len(LISTE)):
    if str(LISTE[k])[-4:]==".dcm":
        LISTE_DCM.append(LISTE[k])
    if str(LISTE[k])[-6:]=="al.png":
        LISTE_PNG.append(LISTE[k])


#print(LISTE_PNG)

volume_numpy=np.zeros((len(LISTE_PNG),512,512))
for k in range (0,len(LISTE_PNG)):
    image = imageio.imread(os.path.join(r"J:\IA\ODIASP\L3_augment2",LISTE_PNG[k]))
    #image = tf.keras.preprocessing.image.img_to_array(os.path.join(r"J:\IA\ODIASP\L3_augment2",LISTE[k]), data_format="channels_last", dtype="float32")
    volume_numpy[k,:,:]=image
#volume_numpy *= 1/255

        
#volume_numpy=np.zeros((len(LISTE_DCM),512,512))
"""
for k in range (0,len(LISTE_DCM)):
    dicom_file = pydicom.read_file(os.path.join(r"J:\IA\ODIASP\L3_augment2",LISTE[k]),force=True)
    dicom_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    img_orig_dcm = (dicom_file.pixel_array)
    #img_modif_dcm = tf.keras.preprocessing.image.img_to_array(k, data_format=channels_last, dtype=float32)
    volume_numpy[k,:,:]=img_orig_dcm
#volume_numpy *= 1/255
"""


print(np.shape(volume_numpy))


AffichageMulti(volume_numpy, frequence=10, axis=0, FIGSIZE = 40)


# In[ ]:


#volume_numpy = volume_numpy[:,:,:,np.newaxis]
#volPREDICTION = model.predict(volume_numpy)


# In[17]:


#APPLICATION AU TEST
DSTEST= r"J:\IA\ODIASP2\Scaled"

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_directory(DSTEST,
                                                  class_mode=None,
                                                  color_mode = "grayscale",
                                                  #subset = "validation",
                                                  target_size=TARGETSIZE,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False)



STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=modelSegMuscles.predict(test_generator,steps=STEP_SIZE_TEST,verbose=1)


# In[18]:


print(np.shape(pred))
PREDICTION = pred[:,:,:,0]
print(np.shape(PREDICTION))
AffichageMulti(PREDICTION, frequence=10, axis=0, FIGSIZE = 40)


# In[ ]:





# In[29]:


fig, ax = plt.subplots(2,2,figsize=(25,25))
for i in range(2):
    dix=np.random.choice(PREDICTION.shape[0],2)
    ax[i,0].imshow(volume_numpy[dix[0],:,:], cmap='gray')
    #ax[i,0].imshow(y_val[dix[0],:,:,0], cmap='magma', alpha=0.5)
    ax[i,1].imshow(volume_numpy[dix[0],:,:], cmap='gray')
    ax[i,1].imshow(pred[dix[0],:,:,0], cmap='magma', alpha=0.5)
plt.show()

