# CS 4641 Group 39 Project Proposal

## Introduction/Background: 
Car price prediction is a concerning research area regarding a country’s economy as well as an individual’s life. According to the Office for National Statistics, UK, the price of used cars has fluctuated greatly each month in recent years. For example, from January to February 2019, the listing dropped by nearly 2,000,000.[1] Factors including transmission, mileage, fuel types, road tax, mpg and engine size often affect the price of used cars. Without a proper way to define the effects of the factors, it is hard to find out the best time to sell or buy a used car. In this project, we are planning to use a data set of pricing of used cars in the UK to predict the price of used cars in the future.

## Problem definition:
What factor affects the price of Volkswagen in the UK most?
How could we best predict the prices of used cars based on features such as gearbox, miles, fuel, tax, miles per gallon or engine size?


## Methods:

1. To predict the car price, the first method we pick is Linear Regression. We begin our method by understanding different attributes. The car price is the dependent target variable, and other car features are independent variables. A heatmap can clearly show how each pair of variables are correlated. While building the model, we should pay attention to multicollinearity. According to supervised learning, we have car prices as desired output values and other features as input variables. We can split the dataset into 70% training data and 30% testing data. By using Linear Regression, we can fit the training data into the model and predict the car prices from the input variables of testing data. Then, we can compare the predicted car price with the actual car price in testing data.

2. The second prediction method we will consider using is Random Forest Regression. The algorithm makes decisions by generating a large number of decision trees. The algorithm is robust and does not overfit. While it generally performs better in classification than regression, its tolerance of incomplete data makes it still worth trying.
The third method we are going to use is Neural Network. It is one of the most famous and widely used machine learning models. Its special way of storing information and neural-like mechanics makes it robust to noises and works with corrupted data.



## Potential results and Discussion:
We will come up with several models to predict the price of a used Volkswagen based on the factors such as transmission, mileage, fuel types, road tax, mpg and engine size. To evaluate the quality of our predictive models, we will compare the predicted results to the actual dataset. We will use an adjusted R square to report the overall model fit, this method also takes the overfitting problem into consideration. In addition, we will use mean squared error (MSE) and mean absolute error (MAE) to compare between models. Ultimately, we aim to produce effective algorithms for dealers and buyers to estimate car prices based on historical values.

## References:
[1] Moran, H.S.and D. (2022) Using auto trader car listings data to transform Consumer Price Statistics, UK, Using Auto Trader car listings data to transform consumer price statistics, UK - Office for National Statistics. Office for National Statistics. Available at: https://www.ons.gov.uk/economy/inflationandpriceindices/articles/usingautotradercarlistingsdatatotransformconsumerpricestatisticsuk/2022-06-28 (Accessed: October 3, 2022).  
Gegic, E., Isakovic, B., Keco, D., Masetic, Z., & Kevric, J. (2019). Car price prediction using machine learning techniques. TEM Journal, 8(1), 113-118. doi:https://doi.org/10.18421/TEM81-16.  
Yadav, A., Kumar, E., & Yadav, P. K. (2021). Object detection and used car price predicting analysis system (UCPAS) using machine learning technique. Linguistics and Culture Review, 5(S2), 1131-1147. https://doi.org/10.21744/lingcure.v5nS2.1660.  

## Proposed timeline:
### September 
9.16  
Project team composition  
9.20 - 9.22  
Brainstorm and select a topic for the project; background research and create a timeline for the project  
9.23 - 9.25  
Find data to use and analyze for the model  
9.25 - 10.2  
Discussion conclusion and potential results  
### October
10.2 - 10.7  
Complete the proposal  
10.7 - 10.14  
Clean the data and build the dataset  
10.17 - 10.24  
Decide on the machine learning models to use for the project  
10.25 - 10.31  
Try two approaches and decide on the needed features  
### November 
10.31 - 11.04  
Try to find more approaches and increase the number of features  
11.05 - 11.11  
Complete and submit the midterm report  
11.11 - 11.27  
Test the model and calculate metrics for the model  
### December
11.27 - 12.01  
Begin to write the final report  
12.01 - 12.09  
Revise the final report  

## Dataset:
The cleaned data set contains data on 100,000 Used Cars in the UK from Kaggle.  
https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?select=audi.csv

## Contribution table:

| Member | Responsibilities |
| ----------- | ----------- |
| Luting Wang | Problem definition, github page |
| Yueqiao Chen | Methods |
| Minzhi Wang | Discussion, video editing |
| Jiaxuan Li | Introduction |
| Xingfang Yuan | Methods |

## Github Page: 
https://github.gatech.edu/pages/lwang797/4641_project/

## Video Link:
https://drive.google.com/file/d/13nvBvGS4xZlc1UhfZ3NvkmzPrfhbzIvS/view?usp=sharing

# Midterm Report

## Data Understanding and Visualization:
The Dataset is sourced from Kaggle and contains 100,000 UK Used Cars data with features. We specifically choose the Toyota cars to predict their prices from other features: model, year, transmission, mileage, file type, tax, mpg, and engine size.  
![data_understanding](https://github.gatech.edu/storage/user/58298/files/89b1c2d3-6532-4966-a9a5-958d0a8683ff)
  
![describe](https://github.gatech.edu/storage/user/58298/files/02ac23f4-3983-4911-aabc-b09195d05b2e)
  
The first step is to make all feature representations numerically. In these features, model, transmission, and file type are strings, so we assign the types in each of these three features with a classification number according to the index in its directory.  
![lable_num](https://github.gatech.edu/storage/user/58298/files/c6002815-c87a-4310-9317-3521451448fc)  
We can find the correlation between each pair of features using ".corr()" and extract the correlation between price and one of the other features. Utilizing matplotlib, we can visually see the correlation from the graphs straightforwardly.  


## Feature Selection
We choose to implement the sklearn.feature_selection module, which could either improve estimators’ accuracy scores or to boost their performance on very high-dimensional datasets. In "select(features, rtrain, rtest)" function we defined, SelectKBest is used to remove all but the k highest scoring features. We use features, rmsetrain and rmsetest as parameters in select function, and get the lineplots of features with respect of rmsetrain and rmsetest. Finally, X-test(fitted and transformed) and X_train_trans(transformed) for features including year, transmission, fuelType, and engineSize are printed.  
![RMSE_trainandtest](https://github.gatech.edu/storage/user/58298/files/ab07efbd-5764-416f-b127-6befee5682b1)

## Supervised Method: Linear Regression
The only implemented method now is Linear Regression. We first input the cleaned data into it directly with all 8 features. Then we did feature selection with the elbow method and ran Linear Regression again with less features.
### 8-feature Linear Regression Result:
![LR_all_features](https://github.gatech.edu/storage/user/58298/files/a1a8cd84-bf04-4d75-bd4b-e5d3c725f8ba)  
The regression's accuracy on test data is 78.6%  
We uses R-squared to evaluate the model's performance, and the R-squared values of the data on train and test dataset are:  
Train R-Squared: 0.758975244447372  
Test R-Squared: 0.7825813624329887  
### 4-feature Linear Regression Result:
![LR_4_features](https://github.gatech.edu/storage/user/58298/files/8cace8f2-1217-4907-936d-f1bbbd1af453)  
The regression's accuracy on test data is 78.3%  
The R-squared values of the data on train and test dataset are:  
New_Train R-Squared: 0.766614191040109  
New_Test R-Squared: 0.7649577196569363  
## Linear Regression conclusion:
Applying PCA, we halved the features and from the visualization and R-squared we can see that the performance of the model doesn't change a lot. This implies PCA works for this dataset.  
From the visualized data points, we can see the data distribution is similar to a exponential pattern. This partly explains why linear regression's performance doesn't change a lot and keeps around 80%.  
## Contribution table:

| Member | Responsibilities |
| ----------- | ----------- |
| Luting Wang | Introduction, github page |
| Yueqiao Chen | Data processing and coding |
| Minzhi Wang | Evaluation |
| Jiaxuan Li | Coding |
| Xingfang Yuan | Coding |
