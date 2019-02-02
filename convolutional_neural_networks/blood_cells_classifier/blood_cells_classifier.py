import numpy as np
np.random.seed(1996)

import matplotlib.pyplot as plt

import json
import time
import os

from keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras import optimizers
from keras.regularizers import l2

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from os.path import split as splt 

from sklearn.metrics import confusion_matrix

from glob import glob


# re-size all the images to this
IMAGE_SIZE = [60, 80]

# training config:
epochs = 25
batch_size = 32

# https://www.kaggle.com/paultimothymooney/blood-cells
train_path = '.../blood-cells/dataset2-master/images/TRAIN'
valid_path = '.../blood-cells/dataset2-master/images/TEST'

# useful for getting number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# useful for getting number of classes
folders = glob(train_path + '/*')

# look at an image for fun
img = np.random.choice(image_files)
plt.imshow(image.load_img(img))
plt.title(splt(splt(img)[0])[1])
plt.show()

# define the layers:
reg = 0.5e-3 # l2-penalty
inp = Input(shape=IMAGE_SIZE+[3])

x = Conv2D(filters=32, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), 
  activation='relu')(inp)
# x = BatchNormalization()(x)
# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(filters=32, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), 
  activation='relu')(x)
# x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
  activation='relu')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
  activation='relu')(x)
# x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(filters=128, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
  activation='relu')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(filters=128, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
  activation='relu')(x)
# x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(filters=256, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg),
 activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Flatten()(x)

x = Dropout(0.5)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)

prediction = Dense(len(folders), activation='softmax')(x)

# create a model object:
model = Model(inputs=inp, outputs=prediction)

# using the defined model from this repo:
# from keras.models import model_from_json
# # load json and create model
# json_file = open('blood_cells_classifier_model.json', 'r')
# model_json = json_file.read()
# model = model_from_json(model_json)

# view the structure of the model:
model.summary()

# tell the model what cost and optimization method to use:
adam = optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.99)
model.compile(
  loss='categorical_crossentropy',
  optimizer=adam,
  metrics=['accuracy']
)


# create an instance of ImageDataGenerator
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  preprocessing_function=lambda x: x/255.0
)


# test generator to see how it works and some other useful things

# get label mapping for confusion matrix plot later
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
  labels[v] = k

i = 0
for x, y in test_gen:
  print("min:", x[0].min(), "max:", x[0].max())
  plt.title(labels[np.argmax(y[0])])
  plt.imshow(x[0])
  plt.show()
  i+=1
  if i==5:
    break


# create generators
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)


# fit the model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)


# save the model:
timestamp = int(time.time())

dir_name = '%s-blood_cells_classifier' % timestamp
os.mkdir(dir_name)
# later save the trained model and plots to:
sv_dir = os.path.join(os.getcwd(), dir_name)
os.chdir(sv_dir)
# print('\ncwd:', os.getcwd())

with open('%d-blood_cells_classifier_model.json' % timestamp, 'w') as f:
  d = json.loads(model.to_json())
  json.dump(d, f, indent=4)

# make a final evaluation on the validation set:
score = model.evaluate_generator(valid_generator)
print('final val_loss:', score[0])
print('final val_acc:', score[1])

# save the weights:
weights = model.get_weights()
np.save('%d-blood_cells_classifier_weights-%f' % (timestamp, score[1]), weights)
print('\nThe parameters were successfully saved to:\n' + sv_dir + '\n')



def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm


cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)


# plot some data:

# loss:
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.show()

# accuracies:
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.savefig('accuracies.png')
plt.show()


# confusion matrices:
from util import plot_confusion_matrix
plot_confusion_matrix(
  cm, 
  labels, 
  title='Train confusion matrix', 
  sv_dir=os.path.join(sv_dir, 'train_confusion_matrix.png')
)

plot_confusion_matrix(
  valid_cm, 
  labels, 
  title='Validation confusion matrix', 
  sv_dir=os.path.join(sv_dir, 'valid_confusion_matrix.png')
)
