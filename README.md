## Udacity Data Science Nanodegree Programt - Project Disaster Response Pipeline

### Introduction
This project is part of the Udacity Data Science Nanodegree Program
Video on youtube: https://www.youtube.com/watch?v=QbLVh5GTuJQ


### Project Overview
In this project real data from disaster messages from Appen (https://www.appen.com/, formerly Figure 8) has been analyzed and a model has been built for an API that classifies disaster messages.
We build a data pipeline to prepare the message data from real natural disasters around the world. We build a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender.

The project provides a web app where you can input a text emergency message and receive a classification in different disater categories. During disasters, a large number of emergency messages reach emergency services via social media or electronic devices. Categorizing those messages via machine learning procedures helps disaster response organizations to filter for the most relevant information and to allocate the messages to the relevant local rescue teams.

### Project Descriptions
The project consists of three parts and the datasets:

ETL Pipeline: process_data.py file with python code to create an ETL pipeline.

Build an ETL pipeline (Extract, Transform, Load) to retrieve emergency text messages and their classification from a given dataset. Clean the data and store it in an SQLite database.

ML Pipeline: train_classifier.py file contains the python code to create an ML pipeline.

Divide the data set into a training and test set. Create a sklearn machine learning pipeline using NLTK (Natural Language Toolkit) using Hyperparameter optimization via Grid Search. The ml model uses the AdaBoost algorithm (formulated by Yoav Freund and Robert Schapire) to predict the classification of text messages (multi-output classification).

Web App:
A web application enables the user to enter an emergency message, and then view the categories of the message in real time.

Data The ml model trains on a dataset provided by Figure Eight that consists of 30,000 real-life emergency messages. The messages are classified into 36 labels.


### Installation:
You need python 3 and the following libraries installed to run the project: pandas, re, sys, json, sklearn, nltk, sqlalchemy, sqlite3, pickle, Flask, plotly.

 
### Instructions:

Run the following commands in the project's root directory to set up your database and model.

1) To run ETL pipeline that cleans data and stores in database use:
'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseProject.db'

2) To run ML pipeline that trains classifier and saves the lachine lerning model use:
'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'

3) Run the following command in the app's directory to run the web app:
'python run.py'

4) Go to in your browser and use:
 'http://0.0.0.0:3000/'


### Below are two screenshots of the web app:

![image](https://github.com/user-attachments/assets/53de5a80-6c43-4721-9391-d2959095f1ed)


![image](https://github.com/user-attachments/assets/9fedbdee-a655-4071-9883-13aef6aceb9e)

