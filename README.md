# Disaster Response Pipeline Project

[//]: # (Image References)
[app_img]:./app_img.png
[app0_img]:./app0_img.png
[test_result_img]:./test_result_img.png


## Introduction

This repository contains the completed Disaster Response Pipeline Project, which implements a supervised training model to  classify messages incoming from a disaster area to highlight what is happening and/or what type of response may be needed from first responders. 

When first launched the app should show a histogram of genres pulled from a database of messages used for training/testing the model.

![][app0_img]

After a message is entered, the app will classify the message according to one or more of 36 possible disaster related categories such as *food*, *water*, *medical help*, etc...

![][app_img]


## Training Data

The training data for the model is found in the two files:

 * *./data/disaster_messages.csv* 
 * *./data/disaster_categories.csv*

Messages in the first file and categories in the second file correspond 
through an id tag found in each. A message can correspond to more than one 
category. For example, the following row in *disaster_categories.csv*

```
73,related-1;request-1;offer-0;aid_related-1;medical_help-1;medical_products-0;
search_and_rescue-0;security-0;military-0;child_alone-0;water-0;food-0;shelter-0;clothing-0;
money-0;missing_people-0;refugees-0;death-0;other_aid-1;infrastructure_related-1;transport-0;
buildings-1;electricity-0;tools-0;hospitals-1;shops-0;aid_centers-0;other_infrastructure-0;
weather_related-1;floods-0;storm-1;fire-0;earthquake-0;cold-0;other_weather-0;direct_report-0
```


means messages that have **id** = 73 correspond to (are classified in) the categories: **related**,**request**,**medical_help**,**other_aid**, **infrastructure_related**,**buildings**,**hospitals**, and **weather-related**.



## Web Application Interface
The interface to the model is a Web application accessed simply through a browser. The front end of the application is built using the [Bootstrap](https://getbootstrap.com/) library, and receives data from the model in the back end via the [Flask](https://en.wikipedia.org/wiki/Flask_(web_framework)) framework.   

 
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
   Part of the training set here is reserved for testing the model and the results will be output here. Test results
   are shown for each of the 36 categories. 
   
   ![][test_result_img]
   
   
   The column **support** refers to the number of messages in the test. For the *STORM* category there are 498 messages
   that are classified in this category (1.0) and 4746 messages that are not (0.0). 
   
   **precision** is the ratio of true positives (pred =1;test=1) to all positive (affirmative) predictions (pred = 1;test=
   0,1), whereas
   
   **recall** is the ratio of true positives (pred =1;test=1) to all correct predictions  (pred = test).
   
   The **f1-score** is simply the harmonic mean of the precision and recall, and the 
   
   **accuracy score** is the ratio of correct predictions to the number of messages in the test set (for the given
   category). So for *EARTHQUAKE*, %98.89 of 5244 test messages were predicted correctly to either correspond to this
   category or not. 
   
   We can expect that recall may be fairly low for a category where we have only a relatively small number of 
   in-category messages, because the numerator of the recall , number of true positives, will be small compared 
   to the number of false negatives (pred=0;test=0) in the denominator. This is demonstrated in the *FIRE* category in the
   print out above.
   
   

3. Run the web application with the model running in the back end.

   Run the following command in the app's directory to run the web app.
    `python run.py`

   Navigate your web browser to http://0.0.0.0:3001/ or http://127.0.0.1:3001 (or wherever you call home).

