import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
import pickle


# -----------------------------------------------------------------------------
PATH_to_mnist = "/Users/los_napath/PycharmProjects/openCV/digit_detection/mnist"
countError = 0
images = []
labels = []
s = 0
list_all_digits = os.listdir(PATH_to_mnist)
len_list_all_digits = len(list_all_digits) - 1 # minus dir ".DS_Store"
imageDimension = (32, 32, 1)
batchSizeVal = 3
epochsVal = 10
stepsPerEpochVal = 2000
# -----------------------------------------------------------------------------

# IMPORT IMAGES
print("Importing images....")
for i in range(len_list_all_digits):
    # i -> int from 0 to 9 (name of each digit directory)
    current_path = PATH_to_mnist + "/" + str(i)
    list_each_digit = os.listdir(current_path)

    for j in list_each_digit:
        # j -> string name of each digit image eg. "8808.png"
        current_image_path = current_path + "/" + j
        img = cv2.imread(current_image_path)
        img = cv2.resize(img, (imageDimension[0], imageDimension[1]))
        images.append(img)
        labels.append(i)

    print(i, end=" ")

print()
print("Imported images (SUCCESS)")
print("images(list)", len(images))
print("labels(list)", len(labels))

# print(type(images), type(images[10]))
# print(type(labels))
# print(labels)

# Convert images & labels to numpy array
images = np.array(images)
labels = np.array(labels)
print("images(shape)", images.shape)
print("images[10](shape)", images[10].shape)
# print(type(images), type(images[10]))
# print(type(labels))
# print(labels)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)
print(X_train.shape)
print(X_validation.shape)

# PLOT BAR CHART FOR DISTRIBUTION OF IMAGES
numOfSamples = []
for digit in range(0, len_list_all_digits):
    numOfSamples.append(len(np.where(y_train == digit)[0]))
print(numOfSamples, sum(numOfSamples))

plt.figure(figsize=(10, 5))
plt.bar(range(0, len_list_all_digits), numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


# PREPROCESS ALL THE IMAGES
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

# print(X_train.shape)
# print(X_test.shape)
# print(X_validation.shape)

# Create the depth to the images
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0],
                                    X_validation.shape[1],
                                    X_validation.shape[2], 1)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

# IMAGE AUGMENTATION
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

# ONE HOT ENCODING OF MATRICES
y_train = to_categorical(y_train, len_list_all_digits)
y_test = to_categorical(y_test, len_list_all_digits)
y_validation = to_categorical(y_validation, len_list_all_digits)


# CREATING THE MODEL
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1,
                      input_shape=(imageDimension[0], imageDimension[1], 1),
                      activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len_list_all_digits, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())

print()
print("Starting training process...")
print()


# STARTING THE TRAINING PROCESS
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batchSizeVal),
                                           steps_per_epoch=stepsPerEpochVal,
                                           epochs=epochsVal,
                                           validation_data=(X_validation, y_validation),
                                           shuffle=1)


# PLOT THE RESULTS
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# EVALUATE USING TEST IMAGES
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

print(type(model))
# SAVE THE TRAINED MODEL
# pickle_out = open("model_trained.p", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()

path = "/Users/los_napath/PycharmProjects/openCV/digit_detection"
model.save("trained_model")
print("Model saved.")
