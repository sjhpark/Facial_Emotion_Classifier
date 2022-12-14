# Facial_Emotion_Classifier
**Facial Expression Emotion Recognition and Classification**

This repository contains the codes/scripts for the process to randomly split dataset and train a baseline CNN model. Further, we have implemented scripts to lively detect faces and predict their emotions in 2 ways: 1. Webcam 2. Screen-mirroring.

Facial emotion recognition (FER) is an interesting field, which has several fields to be applied to such as healthcare, human-human, and human-machine interactions. Furthermore, FER is an important aspect of predicting psychological states of interlocutors during social interactions. If a machine can recognize the emotion of the person based on its facial expression, there is a huge potential for the industry and market to take advantage of understanding their consumers’ mental states and thus promote improved user and customer satisfaction.  Here, we trained a baseline CNN with the mini version of AffectNet benchmark and achieved around 70% accuracy on predicting emotions per facial expressions. We also created scripts to do live face detection via OpenCV and predict the detected faces' emotions.

Please find the benchmark dataset (AffectNet) we have used to train our CNN model in the following links:
* http://mohammadmahoor.com/affectnet/
* http://mohammadmahoor.com/wp-content/uploads/2017/08/AffectNet_oneColumn-2.pdf

# Directory Setup
Due to the ownership of the dataset we used (AffectNet) to train our CNN model, we cannot provide the dataset here.
Please create or grab your own face image dataset and place them into "dataset" folder.
The directory tree should look like this:
* src
  * Facial_Emotion_Classification.ipynb
  * current_best_model.pt
  * dataset
  * * extracted_train_tar
  * * * train_set
  * * * * annotations
  * * * * * 0.exp.npy, 1_exp.npy, ...
  * * * * images
  * * * * * 0.jpg, 1.jpg, ...
  
# Important Libraries/Modules Setup
Pytorch 1.13.0:
* pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
* We recommend Pytorch 1.13.0 or above since 1.12.0 or below seems to fail to split the odd number of data.
Tensorflow
* pip install tensorflow
OpenCV
* pip install opencv-python


# Demo - Live Face Detection & Emotion Recognition & Classification

[![Watch the video]()](https://user-images.githubusercontent.com/83327791/207692443-8e0c4d54-7eb0-4343-99a2-35cb1ce5ed92.mp4)


