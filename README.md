# Bati Bank Credit Scoring Model

## Business Need

Bati Bank, a leading financial service provider with over 10 years of experience, is partnering with an upcoming successful eCommerce company to enable a buy-now-pay-later service. This service will allow customers to buy products on credit if they qualify. The goal is to create a Credit Scoring Model using the data provided by the eCommerce platform.

Credit scoring is the process of assigning a quantitative measure to a potential borrower to estimate the likelihood of default in the future. Traditionally, creditors build credit scoring models using statistical techniques to analyze various information of previous borrowers in relation to their loan performance. The model can then be used to evaluate a potential borrower by providing similar information used to build the model. The result is either a score representing the creditworthiness of an applicant or a prediction of whether an applicant will default in the future.

## Project Objectives

1. Define a proxy variable to categorize users as high risk (bad) or low risk (good).
2. Select observable features that are good predictors (have high correlation) of the default variable.
3. Develop a model that assigns risk probability for a new customer.
4. Develop a model that assigns credit score from risk probability estimates.
5. Develop a model that predicts the optimal amount and duration of the loan.

## Data and Features

The dataset provided by the eCommerce platform includes various features related to customer transactions. The key features include:

- CustomerId
- TransactionId
- Amount
- TransactionStartTime
- CurrencyCode
- ProviderId
- ProductId
- ProductCategory
- ChannelId
- PricingStrategy
- FraudResult

## Tasks:

### 1 - Understanding Credit Risk

Focus on understanding the concept of Credit Risk. Key references:

- [Credit Risk Modeling](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
- [Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Credit Scoring Approaches](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [Developing a Credit Risk Model](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
- [Credit Risk](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
- [Credit Risk Management](https://www.risk-officer.com/Credit_Risk.htm)

### 2 - Exploratory Data Analysis (EDA)

1. Overview of the Data
2. Summary Statistics
3. Distribution of Numerical Features
4. Distribution of Categorical Features
5. Correlation Analysis
6. Identifying Missing Values
7. Outlier Detection

### 3 - Feature Engineering

1. Create Aggregate Features
2. Extract Features
3. Encode Categorical Variables
4. Handle Missing Values
5. Normalize/Standardize Numerical Features

### 4 - Default Estimator and WoE Binning

1. Construct a default estimator (proxy)
2. Assign all users the good and bad label
3. Perform Weight of Evidence (WoE) binning

### 5 - Modelling

1. Model Selection and Training
2. Hyperparameter Tuning
3. Model Evaluation

### 6 - Model Serving API Call

1. Create a REST API to serve the trained machine-learning models for real-time predictions.
2. Choose a framework (e.g., Flask, FastAPI, Django REST framework).
3. Load the model.
4. Define API endpoints.
5. Handle requests.
6. Return predictions.
7. Deployment.

## References:

- [Basel II Capital Accord](https://www.bis.org/publ/bcbs107.htm)
- [Weight of Evidence (WoE) and Information Value (IV) Explained](https://pypi.org/project/woe/)
- [Feature Engineering using Xverse](https://pypi.org/project/xverse/)
