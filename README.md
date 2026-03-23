# datapipeline-project
company name : codetech It solutions

Name : Thilakgowda BK

domain name : data science 

Intern ID : CTIS6902 

Duration : 4 weeks 

Description of the task : 
This code builds a complete machine learning workflow using the scikit-learn pipeline framework to preprocess data and train a classification model. It is designed to handle real-world datasets that typically contain both numerical and categorical features, as well as missing values. The goal of the script is to predict a target variable (in this case, "purchased") using a clean, automated pipeline.

The process begins by importing essential libraries such as pandas and NumPy for data handling, and several modules from scikit-learn for preprocessing, modeling, and evaluation. The dataset is loaded from a CSV file into a pandas DataFrame, and a preview of the data is printed to understand its structure. The target column is separated from the feature set, where X contains the input features and y contains the labels.

Next, the code identifies which columns are numerical and which are categorical. This distinction is important because different preprocessing techniques are applied to each type. Numerical features are handled by a numeric pipeline, which first fills in missing values using the mean of each column (SimpleImputer(strategy="mean")), and then scales the data using StandardScaler. Scaling ensures that all numeric features have a similar range, which helps improve the performance of many machine learning models.

Categorical features are processed using a separate categorical pipeline. Missing values in these columns are filled with the most frequent value using SimpleImputer(strategy="most_frequent"). Then, categorical variables are converted into numerical format using OneHotEncoder, which creates binary columns for each category. The parameter handle_unknown="ignore" ensures that the model can handle unseen categories during prediction without errors.

Both pipelines are combined using a ColumnTransformer, which applies the appropriate transformations to the respective columns. This ensures that preprocessing is done consistently and efficiently in a single step.

The transformed data is then passed into a full pipeline that includes both preprocessing and a machine learning model. In this case, a LogisticRegression model is used, which is a common algorithm for binary classification tasks. By integrating preprocessing and modeling into a single pipeline, the code ensures that all steps are applied consistently during both training and prediction.

The dataset is split into training and testing sets using train_test_split, allowing the model to be evaluated on unseen data. The pipeline is trained on the training set, and predictions are made on the test set. The model’s performance is measured using accuracy, which indicates the proportion of correct predictions.

Finally, the code demonstrates how to extract the fully processed dataset after preprocessing. The transformed features are converted into a new DataFrame with meaningful column names and saved to a CSV file. This step is useful for inspecting the processed data or using it in other workflows.

Overall, this script provides a clean, modular, and scalable approach to building machine learning models, making it easy to maintain and extend for more complex tasks.

output of the task:

<img width="1060" height="670" alt="Image" src="https://github.com/user-attachments/assets/9e4cdc56-e67a-4362-a746-50a6e8c9b060" />
