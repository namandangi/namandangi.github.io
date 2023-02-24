---
layout: default
---


<h1 align="center"> Diabetes Prediction </h1>

<h3 align="center"> Team 23 </h3>
<h4 align="center"> Hajin Kim, Naman Dangi, Netra Ghaisas, Hubert Pan, Pooja Pache </h4>


## Introduction:

Diabetes is a serious chronic disease that can lead to reduced quality of life and life expectancy. Type 2 diabetes typically develops in adulthood and can often be managed with lifestyle changes and/or medication. Prediabetes is a condition which entails higher blood sugar levels than normal but not high enough to be classified as type 2 diabetes. The CDC has indicated that as of 2019, 37.3 million Americans have diabetes and 96 million have prediabetes, and 1 in 5 diabetics and 8 in 10 prediabetics are unaware of their risk. 

Complications like cardiovascular disease, vision loss, lower-limb amputation, and kidney disease are associated with chronically high blood sugar levels for those with diabetes. Predictive models can assist medical professionals in identifying patients who pose a high risk of developing diabetes even before symptoms show up, enabling early intervention and treatment to prevent or delay the onset of associated complications.

We plan on using the Diabetes Health Indicators dataset from Kaggle and will be predicting the Diabetes_012 column based on the other columns. The dataset is based on 253,680 survey responses from the CDC's BRFSS2015 survey.

## Problem Definition:

Identifying individuals who have or at high risk for diabetes is a critical component for public health initiatives that deal with diabetes.

For this project we plan on using the following dataset: 
- https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset to:

    - Build models that accurately predict if someone has diabetes or pre-diabetes
    - Identify if there are any features that are more highly correlated with diabetes than others

## Methods:

Initially, data cleaning and preprocessing will be our fundamental tasks. After data transformation of raw data into a usable format, we will split the dataset into a train (80%) and test (20%) set using Python libraries. As part of this process we may sample the data in a way to remedy the class imbalance that is present in the data. Further, we aim to implement five algorithms for: Supervised and Unsupervised learning. We decided to use Logistic Regression, Decision Trees and Random Forest algorithms for supervised learning while, for unsupervised learning, we intend to use K Means clustering and Gaussian Mixture Model.
Additionally, we plan on data visualization, feature exploration, and comparing accuracy scores of the different supervised and unsupervised algorithms we have used for diabetes prediction.

## Potential Results and Discussion:

For the project, we plan to conduct a comparative analysis of the performance of the various models. In order to accomplish this, we will take into consideration a range of potential factors, such as Confusion Matrix, Accuracy, ROC Curves, mutual information scores, and mean squared error.
The specific metrics used for this comparison will be determined by the models chosen for evaluation. In terms of results, it is anticipated that certain machine learning algorithms will demonstrate superior performance in relation to others.
We might need to address class imbalance, duplicates, missing values, data normalization, and determining optimum number of clusters based on further analysis of the dataset.

## References: 

- Teboul, Alex. Diabetes Health Indicators Dataset. https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset. Accessed 23 Feb. 2023.
- CDC - BRFSS - Survey Data & Documentation. 29 Aug. 2022, https://www.cdc.gov/brfss/data_documentation/index.htm.
- Xie Z, Nikolayeva O, Luo J, Li D. Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques. Prev Chronic Dis 2019;16:190109. DOI: http://dx.doi.org/10.5888/pcd16.190109external icon.
- American Diabetes Association. Economic costs of diabetes in the U.S. In 2007. Diabetes Care. 2008 Mar;31(3):596-615. doi: 10.2337/dc08-9017. Erratum in: Diabetes Care. 2008 Jun;31(6):1271. PMID: 18308683.

## Timeline:

<iframe width="700" height="700" frameborder="0" scrolling="no" src="https://gtvault.sharepoint.com/sites/CS7641MLProject309/_layouts/15/Doc.aspx?sourcedoc={98a9cf88-40c2-4ff9-b4a7-cc3d1a590a98}&action=embedview&wdAllowInteractivity=False&wdHideGridlines=True&wdHideHeaders=True&wdDownloadButton=True&wdInConfigurator=True&wdInConfigurator=True&edesNext=true&resen=false&ed1JS=false"></iframe>



