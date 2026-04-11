Clustering-based Detection of Abnormal Weather Patterns in Urban Time Series Data
Hybrid Machine Learning • Time-Series Analytics • Interactive Dashboard • Urban Climate Insights

A complete Data Science pipeline that detects abnormal weather patterns across urban regions using a hybrid anomaly detection framework combining Isolation Forest and K-Means clustering — with fast parquet-based storage and a Streamlit dashboard for real-time exploration.

# Repository Overview

## Project Description
### Urban weather systems are highly dynamic and often experience unexpected anomalies such as:
        - Sudden temperature spikes 
        - Unusual humidity drops 
        - Extreme weather outliers 
Traditional threshold-based methods fail to capture these complex patterns effectively.
### This project builds a hybrid machine learning pipeline that:
    - cleans and preprocesses raw weather time-series data,
    - detects anomalies using unsupervised learning,
    - stores results efficiently,
    - and visualizes insights through an interactive dashboard.


##	Project Highlights
-  Multi-city weather data analysis  
- Hybrid anomaly detection using:
  -> Isolation Forest
  -> K-Means Clustering  
-  Interactive Streamlit dashboard  
- Efficient parquet-based storage  
- Time-series trend visualization  
- Detection of both:
  - >global anomalies
  - >local cluster deviations  
-  End-to-end Data Science workflow  


##	Problem Statement
    Urban weather data contains hidden anomalies that are difficult to identify manually.
    This project aims to:
        - analyze weather trends from multiple cities,
        - identify abnormal weather patterns,
        - improve anomaly detection using hybrid ML,
        - and present findings through intuitive visual analytics.


##	Objectives
    - Preprocess and clean raw weather data  
    - Handle missing values effectively  
    - Detect anomalies using Isolation Forest  
    - Perform clustering using K-Means  
    - Combine both models for better accuracy  
    - Store processed outputs in Parquet format  
    - Build a dashboard for visualization  

##	Why This Project?
    Weather anomalies can impact:
        - urban planning
        - public safety
        - transportation
        - environmental monitoring
    This project treats weather data as a **spatiotemporal behavioral system** rather than simple readings.
    It answers:
    - When do abnormal weather events occur?
    - Which cities show unusual patterns?
    - How severe are these deviations?

##	Project Workflow

Raw CSV Weather Data -> Data Cleaning & Missing Value Handling -> Feature Selection & Standardization -> Isolation Forest (Global Anomalies) -> K-Means Clustering (Local Anomalies) -> Hybrid Anomaly Detection -> Parquet Storage -> Interactive Dashboard

# TECHNOLOGIES USED
     Python
     Data preprocessing: NumPy, Pandas
     Machine Learning: Scikit-learn
     Distance metrics: SciPy
     Dashboard: Streamlit
     Visualisation: Plotly
# PROJECT STRUCTURE
├── app.py                     # Streamlit dashboard
├── preprocess_data.py         # Data cleaning and preprocessing
├── run_pipeline.py            # Offline ML pipeline
├── config.py                  # Project configs

├── data/                      # Raw weather CSV files
├── processed_data/            # Processed parquet files

├── components/
│   ├── charts.py              # Plotly charts
│   ├── sidebar.py             # Filters & controls

├── utils/
│   ├── loader.py              # Data loading helpers
│   ├── clustering.py          # KMeans clustering logic

└── README.md

# METHODOLOGY
   ## 1. Data Preprocessing
### To ensure high-quality input:
Missing values handled using:
Forward fill
 Backward fill
 Mean imputation
 ### Feature selection:
->Temperature
->Humidity
->Pressure
->Wind speed
Standardization for ML models
## 2. Isolation Forest
Isolation Forest is used for global anomaly detection.
### Why?
Efficient for large datasets
Detects outliers without labeled data
Works well on complex distributions
## 3. K-Means Clustering
K-Means identifies clusters of similar weather patterns.
### Cluster-based anomaly logic:
Calculate distance of points from centroids
Points far from centroid → anomaly
## 4. Hybrid Anomaly Detection
Final anomaly is determined by:
final_anomaly = IsolationForest OR KMeans
### Benefits:
Detects both global and local anomalies
More robust than single-model detection

# HOW IT WORKS
## Offline Pipeline
    Run once to:
    preprocess raw CSV data
    detect anomalies
    save parquet files
## Online Dashboard
    Dashboard:
    loads parquet files
    displays trends & anomalies
    provides user interaction
