from classifiers import *
from keras.preprocessing.image import ImageDataGenerator

# 1 - Load model Meso4
classifier = Meso4()
classifier.load('weights/Meso4.h5')

print('=============================')
print('Prediction using Meso4 model')
print('=============================')

# 2 - Minimial image generator
dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training')

def getClass(class_number):
    if class_number==1.0:
        return 'Real'
    else:
        return 'Forged'

# 3 - Predict
counter=1
while(counter<=20):
    X,y = generator.next()
    print('Image: #', counter, 'Predicted Probability:', classifier.predict(X)[0][0], 'Class :', getClass(y[0]), '\n')
    counter=counter+1
     
# 1 - Load model MesoInception4
classifier = MesoInception4()
classifier.load('weights/MesoInception.h5')

print('\n\n\n======================================')
print('Prediction using MesoInception4 model')
print('======================================')

# 2 - Minimial image generator
dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training')

# 3 - Predict
counter=1
while(counter<=20):
    X,y = generator.next()
    print('Image: #', counter, 'Predicted Probability:', classifier.predict(X)[0][0], 'Class :', getClass(y[0]), '\n')
    counter=counter+1