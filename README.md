# Filtered Fuzzy Time Series
This project brings a fuzzy time series model for the task of regression in time series that have great influences of stochastic components. The developed model was compared with the state of the art and achieved superior results. For more information, see the article [Using fuzzy clustering to address imprecision and uncertainty present in deterministic components of time series](xxx).



## Run

The project already comes with a configured environment and can be run with `docker` and `make:`

 -  Clone: `git clone https://github.com/Marcos001/Filtered-Fuzzy-Time-Series.git`
 -  Run: 
     -  `make run` (init container docker with project folder and data)
     -  `make vlogs`
         -  acess URL generate in logs of container docker: `http://127.0.0.1:8888/lab?token=...`

### Project structure

- **F-fts/**
  - **data/**: datasets used
  - **FTS/** : library with created model and state-of-the-art methods
  - **index.R**: Validation index used to estimate amount fuzzy sets 
  - **QE-Grid_Search_Alabama.ipynb**: Grid search in time series enrollments of Alabama
  - 

