import sys
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow
from model.model import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# first lets collect the arguments
inp_dir = ""
out_dir = ""
try:
    inp_dir = sys.argv[1]
    out_dir = sys.argv[2]
except Exception as e:
    print("Lets skip that for now")

data_dir = "../lfw"
i = 0
try:
    for r, d, f in os.walk(data_dir):
        for file in f:
            path_file = os.path.join(r, file)
            shutil.copy2(path_file, "../imagesfaces/faces")
except Exception as e:
    print('Already done')


# got the images dataset

curr_path = "../imagesfaces"
image = os.listdir(curr_path)[0]
# print(image)
imgpath = curr_path + "/" + image
plt.imshow(plt.imread(imgpath))
# plt.show()
# get it ot array

#print("Done.. , Shape of data is", data.shape)
input_shape = (248, 248, 3)


m = Model(input_shape)
mod = m.createmodel()  # creating the model
earlystop = EarlyStopping(monitor='val_loss', patience=4)
train_datagen = ImageDataGenerator(rescale=1. / 255)
# Training on 13000 samples
train_batches = train_datagen.flow_from_directory("curr_path/",
                                                  target_size=(248, 248), shuffle=True, class_mode='input', batch_size=32)
tst_batch = train_datagen.flow_from_directory("test_dir/",
                                              target_size=(248, 248), shuffle=True, class_mode='input', batch_size=32)

hist = mod.fit_generator(train_batches, validation_data=tst_batch, callbacks=[earlystop],
                         steps_per_epoch=train_batches.samples // 32,
                         epochs=60)
