# importing all required libraries
import pandas as pd # library to manage dataframes
import numpy as np  # library for array calculations
import matplotlib.pyplot as plt # library for creating plots 
import seaborn as sns # library for statistical analysis and visualizations
from collections import Counter
from sklearn.preprocessing import StandardScaler # library for normalizing
from sklearn import preprocessing
#from sklearn.model_selection import train_test_split # library for splitting dataset
# url to extract dataset
url='https://archive.ics.uci.edu/ml/machine-learning-databases/00542/log2.csv'
df=pd.read_csv(url) # reading dataset from the url
print(df.head())  # printing first 5 values from dataset


#Calculating count of null Values
print(df.isnull().sum())

# printing shape of dataframe (rows x columns)
print(df.shape) # printing shape of dataframe (rows x columns)

#Types of classes
print(df['Action'].unique()) # printing all unique class labels

# describing dataframe
print(df.describe())

#Count Number of Values Belonging to each class
print(df['Action'].value_counts())

# creating plot of  Number of Values Belonging to each class
sns.countplot(x=df['Action'])

#sns.pairplot(df) 
# creating pairplot of all the features
# creating a Heatmap using correlation matrix
corr=df.corr()
plt.figure(figsize=(14,6))
sns.heatmap(corr,color="k",annot=True)

# list of all features present in dataset
features=['Destination Port','Source Port','NAT Source Port',
          'NAT Destination Port','Bytes','Bytes Sent','Bytes Received',
          'Packets','Elapsed Time (sec)','pkts_sent','pkts_received']
label=['Action'] # label stores class label values 

y = df[label] # storing value of class label
x=df[features] # storing all the values of features

x_val = x.values

# for loop creates distplot of each feature using sns library
for i in range(11):
 sns.distplot(x_val[i]) 
 plt.xlabel(features[i])
 plt.show()
 
# checking which features are not normalized using skewness(positive/negative/zero)
for j in features:
    skew = df[j].skew()
    sns.distplot(df[j], kde= False, label='Skew = %.3f'%(skew), bins=30)
    plt.legend(loc='best')
    plt.show()

# creating box plot to show outliers in all features   
plt.figure(figsize=(10,15))
for i,col in enumerate(list(x.columns.values)):
    plt.subplot(4,3,i+1)
    df.boxplot(col)
    plt.grid()
    plt.tight_layout()

    
# Detecting all the observations with more than four outlier using inter quartile range
def Iqr(df):    
    out_index = []
    for col in df.columns.tolist():
        quartile1 = np.percentile(df[col], 25)
        quartile3 = np.percentile(df[col],75)
        IQR = quartile3 - quartile1
        out_list_col = df[(df[col] > quartile3 + 1.5 * IQR )|(df[col] < quartile1 - 1.5 * IQR) ].index
        out_index.extend(out_list_col)
    out_index = Counter(out_index)
    result = list( k for k, v in out_index.items() if v >4 )
    # taking feature with more than 4 outliers
    return result

print('Number of observations with more than 4 outliers in this dataset are %d'%(len(Iqr(df[features]))))        
     
print(df.info())

# removing outliers from dataset 
out_index = Iqr(df[features])
df = df.drop(out_index).reset_index(drop=True)

print(df.shape) # printing shape of dataset after removing outliers
   
    
print(df.info())    

y = df[label]  # stores class label values
x=df[features] # x stores all features values after removing outliers


# Normalizing the data using Standard Scaler method
scaler=StandardScaler()
x=scaler.fit_transform(x) # normalizing on data (without outliers)  


# creating distplot for each feature after removing outliers from each instance
x2 = x
for i in range(11):
 sns.distplot(x2[i])
 plt.xlabel(features[i])
 plt.show()
 
























#