import csv
import cv2
import numpy as np
import math
import random
from scipy import ndimage

#Keras Module import

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import ELU


lines = [] 
images = [] #Stores images
measurements =[] #Stores steering angle

#Read from CSV file generated from simulator


# Udacity Data  
with open('./data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    first_row = next(reader)
    for line in reader:
        lines.append(line)
        
# My Data for recovery correction

with open('./BRIDGE/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)    


with open('./RECOVERY/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)         

with open('./RECOVERY2/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)             
        
with open('./RECOVERY3/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)   
       
#Reduce samples with steering angle bias to 50% to avoid dominant learning on 0 deg        
sample_0 = []
sample_not_0 = []
        
for line in lines:
    if float(line[3]) == 0:
        sample_0.append(line)
    else:
        sample_not_0.append(line)
    
train_data = random.sample(sample_0, math.ceil(len(sample_0)/2))
train_data = train_data + sample_not_0                         


import sklearn

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#lines_reduced = lines[lines[3]!=0].append(lines[lines[3] == 0].sample(frac=0.2))

train_samples, validation_samples = train_test_split(train_data, test_size=0.2)        

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples) #Shuffles data
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            for batch_sample in batch_samples:
                
                images = [] #Stores images
                measurements =[] #Stores steering angle
                
                for i in range(3):

                    name = './IMG_TRAIN/'+batch_sample[i].split('/')[-1]
                    #print(name)
                    #image = cv2.imread(current_path)
                    image = ndimage.imread(name)
                    measurement = float(batch_sample[3])
                    
                    if(i == 1):
                        measurement = measurement + 0.2
                    elif(i == 2):
                        measurement = measurement - 0.2
                        
                    
                    images.append(image)
                    measurements.append(measurement)        

                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    measurement_flipped = -measurement
                    measurements.append(measurement_flipped) 

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#Early Stopping

es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)

#Model

#NVIDIA AUTONOMOUS TEAM ARCHITECTURE 

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5, 5), strides = (2,2)))
model.add(ELU())
model.add(Conv2D(36, (5, 5), strides = (2,2)))
model.add(ELU())
model.add(Conv2D(48, (5, 5), strides = (2,2)))
model.add(ELU())
model.add(Conv2D(64, (3, 3)))
model.add(ELU())
model.add(Conv2D(64, (3, 3)))
model.add(ELU())
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.20))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


print(model.summary())

#Run 
#Using Mean Square Error
opt = Adam(lr = 0.0001)
model.compile(loss = 'mse', optimizer = opt)

#model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True)
model.fit_generator(train_generator,
            steps_per_epoch=math.ceil((len(train_samples) * 2)/batch_size), # x2 because of augmentation
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=10, verbose=1,  callbacks=[es_callback])

print(model.summary())

model.save('model.h5')
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

