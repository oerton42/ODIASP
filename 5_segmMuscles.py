#!/usr/bin/env python
# coding: utf-8


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






#strategie multiGPU
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],
                                          cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



"""
loss avec des poids, que j'utilise
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
    input_size = (512,512,1) 
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

