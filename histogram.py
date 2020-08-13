import csv
import cv2
import math
import random
import matplotlib.pyplot as plt

lines = []
measurements = []
sample_0 = []
sample_not_0 = []

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

for line in lines:
    if float(line[3]) == 0:
        sample_0.append(line)
    else:
        sample_not_0.append(line)
    
train_data = random.sample(sample_0, math.ceil(len(sample_0)/2))
train_data = train_data + sample_not_0          
        
        
for line in train_data:
    
    for i in range(3):
        
        measurement = float(line[3])
        
        if i == 1:
            measurement = measurement + 0.2
        elif i == 2:
            measurement = measurement - 0.2
            
           
        measurements.append(measurement) 
    
plt.hist(measurements, bins=100)
plt.savefig('Training_Data.png')
