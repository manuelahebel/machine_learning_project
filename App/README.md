# Titanic Survival Prediction App
Streamlit Web App to predict the outcome of titanic passengers based on available attributes.

## Data

The data for the following example is originally from [Kaggle](https://www.kaggle.com/c/titanic/data)
Since the data has been added to the `data/` directory, cloning this repository would suffice.
## Pre-requisites

The project was developed using python 3.6.7 with the following packages.
- Pandas
- Numpy
- Scikit-learn
- Pandas-profiling
- Joblib
- Streamlit
- pathlib

Installation with pip:

```bash
pip install -r requirements.txt
```

## Getting Started
Open the terminal in you machine and run the following command to access the web application in your localhost.
```bash
streamlit run app.py
```

## Run on Docker
Alternatively you can build the Docker container and access the application at `localhost:8051` on your browser.
```bash
docker build --tag app:1.0 .
docker run --publish 8051:8051 -it app:1.0
```
## Files
- DataPreparation.ipynb : Jupyter Notebook with pre-processing step
- app.py : Streamlit App script
- requirements.txt : pre-requiste libraries for the project
- models/ : trained model files objects
- data/ : source data

## Summary


## Acknowledgements

[Kaggle](https://kaggle.com/), for providing the data for the machine learning pipeline.  
[Streamlit](https://www.streamlit.io/), for the open-source library for rapid prototyping.
