# rvwanalyzr


## Downloading data to analyze
* app-store-scripts/fetch-reviews.js downloads reviews under each data/apps/<app id> folder. There will be 10 files which is the max number of pages that app store API allows.

* To collect app store reviews
  Add a folder named matching the app id under app-store-scripts/data/apps folder and run
 
  ```
  $ cd app-store-scripts
  $ node fetch-reviews.js
  ```
 

## Building model
Model buider runs review data through VADER sentiment analyzer, Naive Bayes classifier and LDA topic modeler. It creates csv files containing the results for each in the output folder of the working directory.

* To build model with the data downloaded by fetch-reviews.js at the default location, app-store-scripts/data/apps/
  ```
  $ python review_analyzer/sent_model_builder.py
  ```
* To build model with the data stored at specific folder
  ```
  $ python review_analyzer/sent_model_builder.py <path to the folder containing folders named [app id]>
  ```
