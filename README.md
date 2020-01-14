# Disaster Response Pipeline Project


## Introduction

This repository contains the completed Disaster Response Pipeline Project, which implements a supervised training model to  classify input messages from a disaster to highlight what is happening and/or what type of response may be needed from first responders. The categories are .


When runnning the app should look like...



## Training Data

The training data for the model is found in the two files:

 *  *disaster_messages.csv* 
 *  *disaster_categories.csv*

Messages in the first file and categories in the second file correspond 
through an id tag found in each. A message can correspond to more than one 
category. For example, the following row in disaster_categories 

```
73,related-1;request-1;offer-0;aid_related-1;medical_help-1;medical_products-0;
search_and_rescue-0;security-0;military-0;child_alone-0;water-0;food-0;shelter-0;clothing-0;
money-0;missing_people-0;refugees-0;death-0;other_aid-1;infrastructure_related-1;transport-0;
buildings-1;electricity-0;tools-0;hospitals-1;shops-0;aid_centers-0;other_infrastructure-0;
weather_related-1;floods-0;storm-1;fire-0;earthquake-0;cold-0;other_weather-0;direct_report-0
```


means messages corresponding that have *id* = 73 correspond to (classified by) the categories: *related*,*request*,*medical_help*,*other_aid*, *infrastructure_related*,*buildings*,*hospitals*, and *weather-related*.



## Web Application Interface
The interface to the model is a Web application accessed simply through a browser. The front end of the application is built using the [Bootstrap](https://getbootstrap.com/) library, and receives data from the model in the back end via the [Flask](https://en.wikipedia.org/wiki/Flask_(web_framework)) framework.   


## Running the Application 
 
To get the application up and running, we need to do the following three steps.

1. Clean the data and archive to a database.
   
   Run the process_data.py in the project root directory
   ```shell
	python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```
   

2. Build and train a model based on the data in the database.

   ```shell
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl   
   ```

3. Run the web application with the model running in the back end.

   Run the following command in the app's directory to run the web app.
    `python run.py`

    Go to http://0.0.0.0:3001/


