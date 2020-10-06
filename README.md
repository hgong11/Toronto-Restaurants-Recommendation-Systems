# Restaurant Recommender Project

### Author 

|Name|Student #|Email|
|:---:|:---:|:---:|
|Rui Chen| 6727535| rchen040@uottawa.ca|
|Hang Gong| 300084007| hgong012@uottawa.ca|
|Minggang Zhou| 300146258 | mzhou064@uottawa.ca|

## Overview

This project is for restaurant recommender system based on user reviews. 

## Dependencies & Speficiation
* Python: Python 3.8 from Anaconda
* Please set PYTHONUNBUFFERED=1 or run python3 -u main.py to start unbuffered mode
* The following need to be added to .local
    * nltk
    * sklearn
    * pandas
    * geopy
    * matplotlib

## Run
Execute "make run"

## Usage
* In the chat box: there are 3 modes
    * New user mode:
        * Enter "NewUser: <text>": add requirments in text such as "I'd like to have Italian food for dinner with nice view and free parking. "
        * This mode will add the requirements in the system and recommend based on the comments above
    * Existing user mode (with Category):
        * Enter "UserId: ***" so that the system could identify your profile
        * Recommended ID: 
            * 6MM9Yqn7UBM8tmpSHQHAAg
            * 7hAhYoMPjHnxKCz6MQ95Bg
            * p6nBKyT9Y_pFJ1WxVEowwA
        * Enter "Category: <category>" to select desired category of food, e.g Italian
    * Existing user mode (without Category):
        * Enter UserId as previous mode
        * Enter "Category: any" to ignore category in restaurant recommendation

## Tools
* data_downloader.py
    * Used to download dataset and store in /dataset
    * Create /dataset if seen error
* data_extractor.py
    * Extract dataset as JSON from original Yelp data and construct new JSONs

## Organization
* Main program: app.py in cosine similarity
* Algorithm of Latent Factor Collaborative Filtering: Latent_Factor_Collabrative_Filtering.py
* HTML in Templates
* CSS and JS in static
* Tools: 
    * data_downloader.py
    * data_extractor.py
* Visualization & Evaluation: 
    * Toronto Restaurants Geo Visualization (1).ipynb
    * Toronto Reviews Analysis and Visualization (1).ipynb

## Reference
* Download and store the raw dataset from Yelp: 
https://www.yelp.com/dataset/documentation/main
* Stored dataset for this project:
https://gofile.io/d/IKXe81
* Backup dataset link: 
https://drive.google.com/drive/folders/1bNojNAUTWnRQgIroFP8EtzKAgH0TCYFY?usp=sharing
