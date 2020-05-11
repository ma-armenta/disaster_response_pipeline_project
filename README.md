# Disaster Response Pipeline Project

### Project Overview
This project consisted of implementing what was learned from the Sessions of Data Engineering: Data extraction, exploration, cleaning, evaluation
and utilization.
Overall the implementation of ETL, NLP and ML pipelines and model trainning to feed data to a portal, and interpret new data comes in through the portal.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/
