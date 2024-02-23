## Team Information

**Team Name:** Trojan Analyst 

**Team Members:**

- Bibhor Acharya (Bibhor.Acharya@trojans.dsu.edu)
- Bijay Shakya (Bijay.Shakya@trojans.dsu.edu)
- James Momoh (James.Momoh@trojans.dsu.edu)
- Kevin Osei-Onomah (Kevin.Osei-Onomah@trojans.dsu.edu)
- Soniya Shahi (Soniya.Shahi@trojans.dsu.edu)

**Institution/Organization:** Dakota State University

## Instructions

Our project folder contains the following files:
- **'README.md':** 	This file contains all necessary instructions to successfully run this project.
- **'Project.ipynb':** 	This is the Jupyter Notebook file showing all the codes.
- **'Project.json':** This is our final model file, which can be directly loaded for predictions on the validation set.
- **'Predicted Dataset':** This `.csv` file contains the prediction of our best model on the validation dataset (20%) masked as `'Fraud YN'`.
- **Link to Predicted Dataset:** 	__http://tinyurl.com/57kze9bj__ 

## Requirements

The following libraries were used for our project and will need to be installed to successfully run the project file.

- Pandas
- Matplotlib
- Seaborn
- XGBoost
- LightGBM
- scikit-learn
- imblearn

### Instructions

```bash
  pip install pandas scikit-learn matplotlib seaborn xgboost lightgbm
```
**OR**
```bash
  conda install pandas scikit-learn matplotlib seaborn xgboost lightgbm
```
**Install `imblearn` package**

```bash
  pip install imbalanced-learn
```
**OR**
``` bash
  conda install -c conda-forge imbalanced-learn
```
## Selected Problem
**Problem #2:** Classification of pattern recognition in the Health Plan ACA claims data for fraud assessment.

## Methodology

### Data Description

We worked on the second project -  “Classification of pattern recognition in the Health Plan ACA claims data for fraud assessment”.  

Original Dataset Link - __http://tinyurl.com/mtnbbfch__ 

It contains `2,374,928 data points`

The dataset contains `88 features`.

- **Independent Features**
  
  `'LOB', 'LOB Description', 'Market Segment', 'Alt Member Number',
  'PAT_MRN_ID', 'Coverage ID', 'Plan Group ID', 'Corporation',
  'Plan Group Name', 'Gender', 'Birth Date', 'Member Age',
  'Member Age Group', 'Death Date', 'Member City', 'Member State',
  'Member Zip Code', 'Claim ID', 'Claim Line', 'Service Date',
  'Service Year Month', 'Received Date', 'Paid Date', 'Paid Year Month', 'Paid Month', 'Billed Amount', 'Allowed Amount', 'Contract Allowed Amount', 'Paid Amount', 'Discount Amount', 'Deductible Amount', 'Coinsurance Amount', 'Copayment Amount', 'Patient Portion', 'Total Allowed Amount', 'Total Paid Amount', 'Quantity', 
  'Network Status', 'Member Network Name', 'Claim Form Type', 'Claim Type', 
  'Major Category', 'Minor Category', 'Place of Service Code', 'Place of Service Type', 'Facility Status', 'Vendor ID', 'Vendor NPI', 'Vendor Tax ID', 'Hospital or Clinic Name','Place of Service City', 'Place of Service Zip Code', 'Place of Service State', 'Provider ID', 'Provider NPI', 'Provider Name', 'Provider Specialty', 'ICD10 Code', 'Primary Diagnosis', 'DRG Number', 'DRG Name', 'Procedure Code', 'Procedure Name', 'Revenue Code', 'Revenue Code Name', 'Modifiers', 'DRG Pricing YN', 'Method to Pay Code', 'Method to Pay', 'Contract ID', 'Contract Name', 'Pricing ID', 'Pricing Type', 'Referral Status', 'Referral Type', 'Referral ID', 'Referral Reason', 'Referral Date', 'Referring Provider', 'Referring Provider Type', 'Referring Provider Specialty', 'Referring Provider Service Area', 'Member Service Area', 'Provider Service Area', 'Health System', 'True Plan', 'Service Type'`
    
- **Target Feature**
`Fraud YN`

### Data Processing Pipeline

- The raw dataset was first inspected to identify any redundant or empty columns.
- From this inspection, 34 empty columns were removed, as they contained no information.
- Additionally, 11 redundant columns were removed that did not provide any extra information beyond existing columns.
- After removing uninformative columns, the cleaned dataset was split into training (80%) and validation (20%) sets for modeling.
- The split was stratified based on the target variable `Fraud YN` to ensure a balanced distribution of the target classes in both training and validation sets. Stratification reduces sampling bias and ensures the model sees a representative sample during training.
- Categorical variable encoding was only applied to the training set, to avoid leaking information to the validation set.
  - For any integer or float columns representing categorical data, these were first converted to object types before further processing.
  - Missing values in categorical features were imputed with a new category label such as `'Missing'`. This allowed the model to learn that missing values   can provide information about a given row.
- Categories in each categorical feature were retained if they have a `'Yes'` count; for features with more than 7 such categories, only those with more than 20 `'Yes'` counts and a `'Yes'` percentage above 60% are kept to focus on highly predictive categories.
- For each column categories that did not belong to the above group were set to `'Remaining'`.
- Mean imputation for the missing values in the `'Quantity'` column.
- Generated 3 new features by calculating differences in days between 4 date features using reference as `'Received Date'`.
- Addressed missing values in the 'Referral Date' feature using the mean date for missing dates and created another column named `'Referral Date_missing'`.
 - Performed `One Hot Encoding` for Categorical features and converted 42 columns to 589 features. 

### Feature Selection

- After performing `One Hot Encoding (OHE)` on categorical variables, there were about `589 features`, so feature selection was necessary to make the model better and avoid overfitting.
- To address a class imbalance in the target variable, we applied the `Synthetic Minority Over-sampling Technique (SMOTE)` solely to the training set.
- `LightGBM (LGBM)` was used to assess feature importance, retaining features with values exceeding a threshold of 5 for further analysis.
- By choosing features with high-importance values, the dataset's size was reduced, resulting in a simpler model that still captured important predictive information. This selected feature set was then used for model training and validation.

### Model Development  and Prediction

**Model Training and Validation**

- We trained two models: `LGBM` and `XGBoost` on our same pre-processed training dataset (80%). 
- For Hyperparameter tuning, we used `GridSearch/Optuna` to select the best hyperparameter values.

**Best Hyperparameter values**

<table>
  <tr>
    <th>eval_metric</th>
    <th>learning_rate</th>
    <th>max_depth</th>
    <th>N_estimators</th>
    <th>objective</th>
  </tr>

<tr>
    <td>logloss</td>
    <td>0.2</td>
    <td>7</td>
    <td>200</td>
    <td>binary:logistic</td>
</tr>
</table>

- We evaluated the performance of both the models in our validation set (20%) using the best parameter values.
- We evaluated our models in terms of evaluation metric: `AUC score`
    - Evaluation result for training set: 
        - `AUC Score: {}`
    - Evaluation result for test set: 
        - `AUC Score: {}`
- Based on our evaluation metric `AUC Score`, we selected the best model: `{model}` for our prediction.

**Prediction Result**

<table>
  <tr>
    <th> Fraud </th>
    <td>{count}</td>
  </tr>
  <tr>
    <th> Non-Fraud </th>
    <td>{count}</td>
  </tr>
</table>
