# Disaster Response Pipeline Project

The project uses disaster data from Figure Eight to classify messages in a number of disaster-related categories.
The training set is heavily imbalanced and classifying in all categories at once results in a low recall for the less frequent ones.
To tackle this, the classifier implemented here is trained and classifies each message in three sequential steps:

1. Classifies the messages in the related category
2. Filters on related messages and classifies in the six main types: aid, weather, infrastrucutre, request, offer, direct_report
3. Filters on aid, weather and infrastructure categories separately
        and classifies in subtypes.

Each step employs its own classifier with hyper-parameters tuned to maximize the f1-macro score, ensuring both satisfactory precision and recall in as many categories as possible.

A live version of the pipeline is hosted at: https://disaster-response-pond.herokuapp.com

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
