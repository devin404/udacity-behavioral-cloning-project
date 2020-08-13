<<<<<<< HEAD
# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

=======
Clone driving behaviour using Deep Learning
===


With this behaviour cloning project, we give steering & throttle instruction to a vehicle in a simulator based on receiving a centre camera image and telemetry data. The steering angle data is a prediction for a neural network model trained against data saved from track runs I performed.
![simulator screen sot](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/images/Self_Driving_Car_Nanodegree_Program.png)

The training of the neural net model, is achieved with driving behaviour data captured, in training mode, within the simulator itself. Additional preprocessing occurs as part of batch generation of data for the neural net training.

##Model Architecture

I decided to as closely as possible use the [Nvidia's End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model. I diverged by passing cropped camera images as RGB, and not YUV, with adjusting brightness and by using the steering angle as is. I experimented with using 1/r (inverse turning radius) as input but found the values were too small (I also did not know the steering ratio and wheel base of the vehicle in the simulator).

Additional experimentation occurred with using [comma.ai](http://comma.ai/), [Steering angle prediction model](https://github.com/commaai/research/blob/master/train_steering_model.py) but the number of parameters was higher then the nvidia model and it worked off of full sized camera images. As training time was significantly higher, and initial iterations created an interesting off road driving experience in the simulator, I discontinued these endeavours.

The model represented here is my implementation of the nvidia model mentioned previously. It is coded in python using keras (with tensor flow) in model.py and returned from the build_nvidia_model method. The complete project is on github here [Udacity Behaviour Cloning Project](https://github.com/hortovanyi/udacity-behavioral-cloning-project)

###Input

The input is 66x200xC with C = 3 RGB color channels.

###Architecture
**Layer 0: Normalisation** to range -1, 1 (1./127.5 -1)

**Layer 1: Convolution** with strides=(2,2), valid padding, kernel 5x5 and output shape 31x98x24, with **elu activation** and **dropout**

**Layer 2: Convolution** with strides=(2,2), valid padding, kernel 5x5 and output shape 14x47x36, with **elu activation** and **dropout**

**Layer 3: Convolution** with strides=(2,2), valid padding, kernel 5x5 and output shape 5x22x48, with **elu activation** and **dropout**

**Layer 4: Convolution** with strides=(1,1), valid padding, kernel 3x3 and output shape 3x20x64, with **elu activation** and **dropout**

**Layer 5: Convolution** with strides=(1,1), valid padding, kernel 3x3 and output shape 1x18x64, with **elu activation** and **dropout**

**flatten** 1152 output

**Layer 6: Fully Connected** with 100 outputs and **dropout**

**Layer 7: Fully Connected** with 50 outputs and **dropout**

**Layer 8: Fully Connected** with 10 outputs and **dropout**

dropout was set aggressively on each layer at .25 to avoid overtraining
###Output

**Layer Fully Connected** with 1 output value for the steering angle.

###Visualisation
[Keras output plot (not the nicest visuals)](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/model.png)

##Data preprocessing and Augmentation
The simulator captures data into a csv log file which references left, centre and right captured images within a sub directory. Telemetry data for steering, throttle, brake and speed is also contained in the log. Only steering was used in this project.

My initial investigation and analysis was performed in a Jupyter Notebook [here](https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/behavorial-cloning-initial-data-exploration.ipynb).

Before being fed into the model, the images are cropped to 66x200 starting at height 60 with width centered - [A sample video of a run cropped](https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/simulator_run1.mp4?raw=true).

![Cropped left, centre and right camera image](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/images/3cameras.png)

As seen in the following histogram a significant proportion of the data is for driving straight and its lopsided to left turns (being a negative steering angle is left) when using data generated following my conservative driving laps.
![Steering Angle Histogram](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/images/steering_histogram.png)

The log file was preprocessed to remove contiguous rows with a history of >5 records, with a 0.0 steering angle. This was the only preprocessing done outside of the batch generators used in training (random rows are augmented/jittered for each batch at model training time).

A left, centre or right camera was selected randomly for each row, with .25 angle (+ for left and - for right) applied to the steering.

Jittering was applied per [Vivek Yadav's post ](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0) to augment data. Images were randomly transformed in the x range by 100 pixels and in the y range by 10 pixels with 0.4 per xpixel adjusted against the steering angle. Brightness via a HSV (V channel) transform (.25 + a random number in range 0 to 1) was also performed.
![jittered image](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/images/jittered_center_camera.png)

During batch generation, to compensate for the left turning, 50% of images were flipped (including reversing steering angle) if the absolute steering angle was > .1.

Finally images are cropped per above before being batched.

###Model Training

Data was captured from the simulator. I drove conservatively around the track three times paying particular attention to the sharp right turn. I found connecting a PS3 controller allowed finer control then using the keyboard. At least once I waited till the last moment before taking the turn. This seems to have stopped the car ending up in the lake. Its also helped to overcome a symptom of the bias in the training data towards left turns. To further offset this risk, I validated the training using a test set I'd captured from the second track, which is a lot more windy.

####[Training sample captured of left, centre and right cameras cropped](https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/simulator_run1.mp4?raw=true)
<video width="960" height="150" controls>
  <source src="https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/simulator_run1.mp4?raw=true">
</video>

Center camera has the steering angle and 1/r values displayed.

####[Validation sample captured of left, centre and right cameras cropped](https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/simulator_runt2.mp4?raw=true)
<video width="960" height="150" controls>
  <source src="https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/simulator_runt2.mp4?raw=true">
</video>
Center camera has the steering angle and 1/r values displayed.

The Adam Optimizer was used with a mean squared error loss. A number of hyper-parameters were passed on the command line. The command I used looks such for a batch size of 500, 10 epochs (dropped out early if loss wasn't improving), dropout at .25 with a training size of 50000 randomly augmented features with adjusted labels and 2000 random features & labels used for validation

```
python model.py --batch_size=500 --training_log_path=./data --validation_log_path=./datat2 --epochs 10 \
--training_size 50000 --validation_size 2000 --dropout .25
```


###Model Testing
To meet requirements, and hence pass the assignment, the vehicle has to drive around the first track staying on the road and not going up on the curb.

The model trained (which is saved), is used again in testing. The simulator feeds you the centre camera image, along with steering and throttle telemetry. In response you have to return the new steering angle and throttle values. I hard coded the throttle to .35. The image was cropped, the same as for training, then fed into the model for prediction giving the steering angle.

```python

steering_angle = float(model.predict(transformed_image_array, batch_size=1))
throttle = 0.35
```

####Successful run track 1
[![Successful run track 1](http://img.youtube.com/vi/aLrV8UMqzKo/0.jpg)](http://www.youtube.com/watch?v=aLrV8UMqzKo)
####Successful run track 2
[![Successful run track 2](http://img.youtube.com/vi/sW2D1T3ev-k/0.jpg)](http://www.youtube.com/watch?v=sW2D1T3ev-k)

note: the trained model I used for the track 1 run, is different to the one used to run the simulator in track 2. I found that the data I originally used to train a model to run both tracks, would occasionally meander on track 1 quite wildly. Thus used training data to make it more conservative to meet requirements for the projects.
>>>>>>> 7c24e6685a7df8822c938ad112da316cc76898aa
