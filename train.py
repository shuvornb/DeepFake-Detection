import numpy as np
from classifiers import *
from pipeline import *
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# 1 - Load the model and its pretrained weights
classifier = Meso4()

#classifier.load('weights/Meso4_DF')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

#2 train the model with total 12225 images 
# images are obtained from both original and deepfake videos of size 4GB
testdataGenerator = ImageDataGenerator(rescale=1./255)
test_generator = testdataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training')
dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'train_images',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training')
total_epochs = 50
epoch_no = 0
acc_train = []
acc_val = []
while(epoch_no < total_epochs):
    itr = generator
    itr2 = test_generator
    index = 0
    acc_count = 0
    acc_test_count = 0
    total_imgs = 12225
    while(index<total_imgs):
        x,y = itr.next()
        classifier.fit(x, y)
        index += 1
    itr = generator
    index = 0
    total_imgs = 12225
    while(index<total_imgs):
        x, y = itr.next()
        index +=1
        y_train_pred =classifier.predict(x)
        if(y_train_pred <0.5):
            y_train_pred_class = 0
        else:
            y_train_pred_class = 1
        if(y_train_pred_class == y):
            acc_count += 1 
    total_test_imgs = 4454
    index = 0
    while(index<total_test_imgs):
        x, y = itr.next()
        index +=1
        y_test_pred =classifier.predict(x)
        if(y_test_pred <0.5):
            y_test_pred_class = 0
        else:
            y_test_pred_class = 1
        if(y_test_pred_class == y):
            acc_test_count += 1 
    epoch_no += 1
    print("Epoch No:  ",epoch_no)
    acc_tr = (acc_count/total_imgs)*100
    acc_test = (acc_test_count/total_test_imgs)*100
    acc_train.append(acc_tr)
    acc_val.append(acc_test)
    print("Accuracy :",acc_tr)
    print("val_accuracy :",acc_test)
figure=plt.figure()
plt.plot(acc_train)
plt.plot(acc_val)
plt.title('Accuracy of Deepfake detection using Meso Inception Net')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'])
plt.savefig('Accuracy.png')
plt.show()

#saving weights into file under weights folder
classifier.save('newweights/Meso4.h5')

'''
# 4 - Prediction for a video dataset

predictions = compute_accuracy(classifier, 'test_videos')
for video_name in predictions:
    print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
'''