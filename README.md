# Facial_Emotion_Classifier
**Facial Expression Emotion Recognition and Classification**

This repository contains the codes/scripts for our paper [https://www.notion.so/6550214df2c14f37b9a3da12a1255e77?v=1ab2e2898277432cb9c2052c1fe05fed](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/1efd441b-ab16-48d9-a925-7fb7569b8f5e/24787_Final_Report_Human_Emotion_Classification.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T034315Z&X-Amz-Expires=86400&X-Amz-Signature=4c038fbaf4b79bf2e9b6078f9f95fe280955ec0c1c2104b144080476b4383623&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%2224787_Final_Report_Human_Emotion_Classification.pdf%22&x-id=GetObject). 

We have created lines of codes to create and train a baseline CNN model and check its accuracy on AffectNet8, the mini-version of human face image benchmark dataset.
Further, we have implemented scripts to lively detect faces and predict their emotions in 2 ways: 1. Webcam 2. Screen-mirroring.

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

# Data
![num_classes_train](https://user-images.githubusercontent.com/83327791/208017576-7120a473-f9b7-4f3c-a62a-abe1c6ba35a2.png)

# Model Training & Accuracy
![Baseline_CNN_Trained_Performance](https://user-images.githubusercontent.com/83327791/208017525-50970dd7-be25-4dd6-8db0-690b8d84bb50.png)

# Confusion Analysis
![confusion_matrix_ratio](https://user-images.githubusercontent.com/83327791/208017616-4d2041fd-5656-4ba6-b44a-265cf54f6295.png)
![confusion_matrix_count](https://user-images.githubusercontent.com/83327791/208017618-d2d1fd87-8fdb-4a40-ac98-863f4f8fc78c.png)
![f1_score](https://user-images.githubusercontent.com/83327791/208017643-77fbb2b0-520b-4bc2-a71f-82c041a069e6.png)

# Demo - Live Face Detection & Emotion Recognition & Classification

![disney_emotions](https://user-images.githubusercontent.com/83327791/208017677-3c6a2972-0d93-4b7c-87d1-506793e30e0c.png)

[![Watch the video]()](https://user-images.githubusercontent.com/83327791/207692443-8e0c4d54-7eb0-4343-99a2-35cb1ce5ed92.mp4)


