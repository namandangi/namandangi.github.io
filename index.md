---
layout: default
---


<h1 align="center"> Diabetes Risk Prediction </h1>

<h3 align="center"> Team 23 </h3>
<h4 align="center"> Naman Dangi, Netra Ghaisas, Hubert Pan, Hajin Kim, Pooja Pache </h4>

## Video Presentation
[Presentation link](https://drive.google.com/file/d/1t_1J68w3xckdhbXIvkJ59LCGbe7P8sfG/view?usp=share_link)

## Introduction:

Diabetes is a serious chronic disease that can lead to reduced quality of life and life expectancy. Type 2 diabetes typically develops in adulthood and can often be managed with lifestyle changes and/or medication. Prediabetes is a condition which entails higher blood sugar levels than normal but not high enough to be classified as type 2 diabetes. The CDC has indicated that as of 2019, 37.3 million Americans have diabetes and 96 million have prediabetes, and 1 in 5 diabetics and 8 in 10 prediabetics are unaware of their risk [2]. 

Complications like cardiovascular disease, vision loss, lower-limb amputation, and kidney disease are associated with chronically high blood sugar levels for those with diabetes [3]. Predictive models can assist medical professionals in identifying patients who pose a high risk of developing diabetes even before symptoms show up, enabling early intervention and treatment to prevent or delay the onset of associated complications.

We plan on using the [Diabetes Health Indicators dataset from Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) [1] and will be predicting the Diabetes_012 column based on the other columns. The dataset is based on 253,680 survey responses from the CDC's BRFSS2015 survey.

## Problem Definition:

Identifying individuals who have or at high risk for diabetes is a critical component for public health initiatives that deal with diabetes.

For this project we plan on using the [Diabetes Health Indicators dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) to:

- Build models that accurately predict if someone has diabetes or pre-diabetes
- Identify if there are any features that are more highly correlated with diabetes than others

## Methods:

Initially, data cleaning and preprocessing will be our fundamental tasks. After data transformation of raw data into a usable format, we will split the dataset into a train (80%) and test (20%) set using Python libraries. As part of this process we may sample the data in a way to remedy the class imbalance that is present in the data. Further, we aim to implement five algorithms for: Supervised and Unsupervised learning. We decided to use Logistic Regression, Decision Trees and Random Forest algorithms for supervised learning while, for unsupervised learning, we intend to use K Means clustering and Gaussian Mixture Model.
Additionally, we plan on data visualization, feature exploration, and comparing accuracy scores of the different supervised and unsupervised algorithms we have used for diabetes prediction.

-- add Data pre-processing --

Supervised: Logistic Regression
In our initial data analysis, we applied the logistic regression algorithm to our large dataset and opted for the SAGA (Stochastic Average Gradient Descent) solver instead of the default LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) solver. This decision was based on SAGA's support for L1 regularization, which can help with feature selection and reducing overfitting. However, we encountered overfitting and to address it, we employed both SMOTE oversampling and Stratified K-fold undersampling techniques. To further enhance our model's accuracy, we utilized PCA for dimensionality reduction. We experimented with different combinations of the number of PCA components and the number of folds, monitoring their impact on recall, accuracy, roc score and explained variance.

Unsupervised: KMeans
As part of the initial data exploration we also ran the KMeans algorithm on the dataset to explore how well the data was partitioned. This was done via the “elbow method” by repeatedly running KMeans multiple times with different numbers of clusters, evaluating those clusters against a set of metrics, and then looking at how increasing the target cluster count affected the measured cluster evaluation scores by graphing the results. For the purposes of this study we looked at the following cluster evaluation scores: purity, weighted f1, and silhouette across the geometric sequence from 3 to 100.
Initially, data cleaning and preprocessing will be our fundamental tasks. After data transformation of raw data into a usable format, we will split the dataset into a train (80%) and test (20%) set using Python libraries. As part of this process we may sample the data in a way to remedy the class imbalance that is present in the data. Further, we aim to implement five algorithms for: Supervised and Unsupervised learning. We decided to use Logistic Regression, Decision Trees and Random Forest algorithms for supervised learning while, for unsupervised learning, we intend to use K Means clustering and Gaussian Mixture Model.
Additionally, we plan on data visualization, feature exploration, and comparing accuracy scores of the different supervised and unsupervised algorithms we have used for diabetes prediction.




## Potential Results and Discussion:

For the project, we plan to conduct a comparative analysis of the performance of the various models. In order to accomplish this, we will take into consideration a range of potential factors, such as Confusion Matrix, Accuracy, ROC Curves, mutual information scores, and mean squared error.
The specific metrics used for this comparison will be determined by the models chosen for evaluation. In terms of results, it is anticipated that certain machine learning algorithms will demonstrate superior performance in relation to others.
We might need to address class imbalance, duplicates, missing values, data normalization, and determining optimum number of clusters based on further analysis of the dataset.For the project, we plan to conduct a comparative analysis of the performance of the various models. In order to accomplish this, we will take into consideration a range of potential factors, such as Confusion Matrix, Accuracy, ROC Curves, mutual information scores, and mean squared error.

 -- Write about data preprocessing results: --

During the data preprocessing stage, we did not detect any null values, but we did come across missing values and duplicates.

Supervised: Logistic Regression:
After evaluating the performance of the Logistic regression algorithm on our dataset, we discovered a class imbalance issue where the recall value for the Pre Diabetes class was zero. To address this issue, we applied the SMOTE oversampling and Stratified K-fold undersampling techniques separately, but both techniques individually resulted in higher accuracies but lower recall and precision metrics for the Pre Diabetes class. A combination of both techniques demonstrated better results, although the accuracies were not optimal. Additionally, we observed that varying the number of PCA components affected the accuracy and roc score, with the Non-diabetic and diabetic classes achieving the highest scores when the number of PCA components equaled the total number of features. The explained variance also increased as the number of PCA components increased. However, utilizing all components can be computationally expensive and may not produce an optimal model. In addition, modifying the number of folds in the Stratified K-fold undersampling technique had a minor effect on accuracy. However, increasing the number of folds can lead to increased computational complexity, prolonged processing times, and greater sensitivity to noise in the data. Consequently, based on the performance metrics of Logistic regression, we concluded that it is not the most effective supervised learning algorithm for our diabetes dataset, and we intend to investigate other algorithms for improved performance.

[Supervised Result Image]

<img src="https://raw.githubusercontent.com/namandangi/namandangi.github.io/main/static/logisticreg/logistic1.png">

Unsupervised: KMeans
The metric used to evaluate the clustering results did not show a significant improvement as the number of clusters increases. In addition, the confusion matrix reveals that all predicted clusters contain mostly points belonging to the Diabetes_012 class, indicating that the clustering algorithm is not able to identify meaningful patterns or clusters in the data. This observation suggests that a different clustering algorithm or preprocessing technique may be more suitable for this dataset. For instance, increasing the number of initializations for the K Means algorithm with larger values of K may improve the clustering results. By using larger initializations, the algorithm will generate more candidate solutions with different starting points and increase the chance of finding a good local optimum. The expectation is that the algorithm can detect more compact clusters that accurately reflect the actual partitions in the data, resulting in an enhancement of the clustering quality measurement. However, it is important to note that increasing the number of initializations and K can also lead to a longer computational time and higher memory usage.


[image here]

## Timeline:

<iframe width="700" height="700" frameborder="0" scrolling="no" src="https://gtvault.sharepoint.com/sites/CS7461MLProjectGroup23/_layouts/15/Doc.aspx?sourcedoc={3313c554-d0e7-4d86-aad0-ed957fcbcf4c}&action=embedview&wdAllowInteractivity=False&wdHideGridlines=True&wdHideHeaders=True&wdDownloadButton=True&wdInConfigurator=True&wdInConfigurator=True&edesNext=true&resen=false&ed1JS=false"></iframe>

(Static Image Alternate)
<img src="https://raw.githubusercontent.com/namandangi/namandangi.github.io/main/static/gantt.png">

## Contribution Table:

<iframe width="700" height="350" frameborder="0" scrolling="no" src="https://gtvault.sharepoint.com/sites/CS7461MLProjectGroup23/_layouts/15/Doc.aspx?sourcedoc={2b953a2b-9772-41aa-99fc-ba5949bab060}&action=embedview&wdAllowInteractivity=False&wdHideGridlines=True&wdHideHeaders=True&wdDownloadButton=True&wdInConfigurator=True&wdInConfigurator=True&edesNext=true&resen=false&ed1JS=false"></iframe>

(Static Image Alternate)
<img src="https://raw.githubusercontent.com/namandangi/namandangi.github.io/main/static/contri.png">

## References: 

1. Teboul, Alex. Diabetes Health Indicators Dataset. [https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). Accessed 23 Feb. 2023.
2. CDC - BRFSS - Survey Data & Documentation. 29 Aug. 2022, [https://www.cdc.gov/brfss/data_documentation/index.htm](https://www.cdc.gov/brfss/data_documentation/index.htm).
3. American Diabetes Association; Economic Costs of Diabetes in the U.S. in 2007. Diabetes Care 1 March 2008; 31 (3): 596–615. [https://doi.org/10.2337/dc08-9017](https://doi.org/10.2337/dc08-9017)
4. Xie Z, Nikolayeva O, Luo J, Li D. Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques. Prev Chronic Dis 2019;16:190109. DOI: [http://dx.doi.org/10.5888/pcd16.190109](http://dx.doi.org/10.5888/pcd16.190109) 


