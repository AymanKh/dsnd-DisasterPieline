# Disaster Response Pipeline Project

### Summary:
This project aims to classify emergency calls into categories. This analysis is done through using text-processing techniques to extract features from text data.

### Data:
the dataset provided contains messages from 36 different categories. The goal is to build a classifier that trains from these messages and predict the class for any given message

### Files:
- process_data.py: this file pre process the text data and apply cleaning strategies to prepare the data for ML model. It also stores the data in the databse
- train_classifier.py: this file reads the database and applies ML model to the data. Also, it evaluates the model and performs grid search to finetune the model
- run.py: this file runs a GUI on the browser that has visualizations and user-input. It also has Flask networking module 
- .html files: these files have the HTML code needed to run the web application


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python3 run.py`

3. Go to http://0.0.0.0:3001/