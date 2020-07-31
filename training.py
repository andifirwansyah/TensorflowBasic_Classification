import sys
import os
from tensorflow import keras
import time

start = time.time()

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
    DEV = True

if DEV:
    epochs = 2
else:
    epochs = 30

train_data_path = 'data/train'
validation_data_path = 'data/test'

"""
Parameters
"""

img_width, img_height = 150, 150
batch_size = 32
samples_per_epoch = 240
validation_steps = 30
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 2
lr = 0.0004

model = keras.Sequential([
    keras.layers.Conv2D(nb_filters1, conv1_size, conv1_size, input_shape=(img_width, img_height, 3)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(pool_size, pool_size)),

    keras.layers.Conv2D(nb_filters2, conv2_size, conv2_size),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(pool_size, pool_size)),

    keras.layers.Flatten(),
    keras.layers.Dense(256),
    keras.layers.Activation("relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(classes_num, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])


train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


"""
Show tensorboard log
"""
log_dir = './tf-log/'
tb_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit_generator(
    train_generator,
    steps_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps=validation_steps)

target_dir = './models/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

model.save('./models/model.h5')
model.save_weights('./models/weights.h5')

#Calculate execution time
end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")