# Sentiment Analysis Pipeline

## JJC257

## Requirements

This project was developed using Python 3.11.0 64-bit and to set up the database locally will require users to install PostgreSQL (in this instance, version 14 was used). 

## Executing the Pipeline

To execute the pipeline, follow the below step-by-step guide. It is recommended that each module is executed in the listed order so that prerequisite data files are available for certain modules to run.

## TWFYScraper

1. Open and run twfy_scraper.py 

## TWFYPreprocessor

1. Open and run twfy_preprocessor.py

## SAModel

1. Open and run preprocessing.py
2. Once preprocessing.py has completed execution, open and run modelling.py
3. Once modelling.py has completed execution, open and run prediction.py

## PostgreSQL Database

1. After installing and logging in to PSQL, as a superuser (postgres) role, create a new database using the below statement:

	CREATE DATABASE parliasent
		WITH
		OWNER = postgres
		ENCODING = ‘UTF8’
		CONNECTION LIMIT = -1;

2. Connect to the database using the below command:

	\c parliasent postgres

3. Create a table within the database using the below statement:

	CREATE TABLE public.predicted_data
	(
		index integer NOT NULL,
		id text,
		speakername text,
		url text,
		vote text,
		date date,
		sentiment_score real,
		overall_sentiment text,
		speech_text_new text,
		nb_prediction text,
		lr_prediction text,
		sgd_prediction text,
		PRIMARY KEY (index)
	);

4. Input the following command – be sure to insert the file path to the SAModel’s predicted_data.csv where appropriate, ensuring the file is located on the same drive as your PostgreSQL installation:

	COPY predicted_data(index, id, speakername, url, vote, date, sentiment_score, overall_sentiment, speech_text_new, nb_prediction, lr_prediction, sgd_prediction)
	FROM '{LOCAL FILEPATH FOR PREDICTED_DATA.CSV}’
	DELIMITER ','
	CSV HEADER;

5. If the import was successful, PSQL will output “COPY” followed by the number of entries imported.

## SAWebApp

1. Open web_app.py
	a. Modify the _DATABASE_URI constant so that it contains your PostgreSQL login details as well as your local host address; for example, "postgresql://postgres:password@127.0.0.1:5432/predicted_data"
2. Run web_app.py
3. Click on the link to the website in the terminal (http://127.0.0.1:5000 by default).
4. Navigate the website.
	a. Click Data to view all available data in the database.
	b. Click Search and input an MP’s name to view only that MP’s prediction results.
