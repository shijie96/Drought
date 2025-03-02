# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE  #(Recursive Feature Elimination) is a technique used to select the most important features for a machine learning model by recursively eliminating the least important ones
from imblearn.over_sampling import SMOTE # Synthetic minority over-sampling technique si a technique used to handle imbalanced datasets by generating synthetic samples for the minority class instead of simply duplicating existing ones.


# Reading the Input data
drought_df = pd.read_csv("C:/Users/shiji/OneDrive/桌面/New folder/Drought/train_timeseries/train_timeseries.csv")
drought_df
drought_df.head()



# WS10M_MIN:	Minimum Wind Speed at 10 Meters (m/s)
# QV2M:	Specific Humidity at 2 Meters (g/kg)
# T2M_RANGE:	Temperature Range at 2 Meters (C)
# WS10M:	Wind Speed at 10 Meters (m/s)
# T2M:	Temperature at 2 Meters (C)
# WS50M_MIN:	Minimum Wind Speed at 50 Meters (m/s)
# T2M_MAX:	Maximum Temperature at 2 Meters (C)
# WS50M:	Wind Speed at 50 Meters (m/s)
# TS:	Earth Skin Temperature (C)
# WS50M_RANGE:	Wind Speed Range at 50 Meters (m/s)
# WS50M_MAX:	Maximum Wind Speed at 50 Meters (m/s)
# WS10M_MAX:	Maximum Wind Speed at 10 Meters (m/s)
# WS10M_RANGE:	Wind Speed Range at 10 Meters (m/s)
# PS:	Surface Pressure (kPa)
# T2MDEW:	Dew/Frost Point at 2 Meters (C)
# T2M_MIN:	Minimum Temperature at 2 Meters (C)
# T2MWET:	Wet Bulb Temperature at 2 Meters (C)
# PRECTOT:	Precipitation (mm day-1)


# Initial data exploration and data cleaning
drought_df.info()

drought_df.dtypes

drought_df.isnull().sum()  # There are 16543883 na values in the column of score
# Check if there are any null value in a single column
drought_df['WS10M_MIN'].isnull().sum()

# Missing values Treatment
# Remove the null values in the target variable as the drought score is only avaiable for once in 7 days.
drought_df = drought_df.dropna()
drought_df.isnull().sum()

# Reformatting the data
drought_df.dtypes

drought_df['Year'] = pd.DatetimeIndex(drought_df['date']).year
drought_df['Month'] = pd.DatetimeIndex(drought_df['date']).month
drought_df['Day'] = pd.DatetimeIndex(drought_df['date']).day
drought_df['score'] = drought_df['score'].round().astype(int)
drought_df.dtypes

print(drought_df['score'])

drought_df['fips'].nunique()
drought_df['score'].nunique()
drought_df['score'].unique()
drought_df['score'].round().value_counts() # count frequency of each unique value

# Exploratory data analysis
# univariate analysis - Descriptive statistics
# Description statistics
drought_df.describe() # For numeric columns
drought_df.describe(include = ['object']) # For categorical columns
print('\nSkewness: \n', drought_df.skew(axis = 0, skipna = True))

column_list = list(drought_df.columns)
column_list

# Univariate Analysis - Distribution of continuous variables
measure_columns_list = ['PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN','WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']
drought_df_measures = drought_df[['PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN','WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']]

for col_name in measure_columns_list:
    plt.figure()
    plt.hist(drought_df_measures[col_name], density = True, bins=30, alpha=0.7)
    plt.xlabel(col_name)
    y_name = 'Density'
    plt.ylabel(y_name)
    plt.grid(True)
    plt.title('Distribution of {col_name}'.format(col_name = col_name))
    plt.show()


# Outlier Treatment
## Identifying Outlier

plt.figure(figsize=(10, 40))
for x in (range(1, 19)):
    plt.subplot(19, 1, x)
    sns.boxplot(x = drought_df_measures.columns[x-1], data = drought_df_measures)
    x_name = drought_df_measures.columns[x-1]
    plt.title(f'Distribution of {x_name}')
plt.tight_layout()

print('Total rows =', len(drought_df_measures.index))
for i in drought_df_measures.select_dtypes(exclude= ['object']).columns:
    print('Number of values beyoud standard outlier limit in', i)
    print(len(drought_df_measures[(drought_df_measures[i] > drought_df_measures[i].mean() + 3 * drought_df_measures[i].std()) | (drought_df_measures[i]< drought_df_measures[i].mean() - 3*drought_df_measures[i].std())]))


# Removing values beyond the standard outlier limit

drought_df = drought_df[(drought_df['PRECTOT'] <= drought_df['PRECTOT'].mean() + 3 * drought_df['PRECTOT'].std()) & (drought_df['PRECTOT'] >= drought_df['PRECTOT'].mean() - 3*drought_df['PRECTOT'].std())]
drought_df = drought_df[(drought_df['PS'] <= drought_df['PS'].mean() + 3*drought_df['PS'].std()) & (drought_df['PS'] >= drought_df['PS'].mean() - 3*drought_df['PS'].std())]
drought_df = drought_df[(drought_df['QV2M'] <= drought_df['QV2M'].mean() + 3*drought_df['QV2M'].std()) &
        (drought_df['QV2M'] >= drought_df['QV2M'].mean() - 3*drought_df['QV2M'].std())]

drought_df = drought_df[(drought_df['T2M'] <= drought_df['T2M'].mean() + 3*drought_df['T2M'].std()) &
        (drought_df['T2M'] >= drought_df['T2M'].mean() - 3*drought_df['T2M'].std())]

drought_df = drought_df[(drought_df['T2MDEW'] <= drought_df['T2MDEW'].mean() + 3*drought_df['T2MDEW'].std()) &
        (drought_df['T2MDEW'] >= drought_df['T2MDEW'].mean() - 3*drought_df['T2MDEW'].std())]

drought_df = drought_df[(drought_df['T2MWET'] <= drought_df['T2MWET'].mean() + 3*drought_df['T2MWET'].std()) &
        (drought_df['T2MWET'] >= drought_df['T2MWET'].mean() - 3*drought_df['T2MWET'].std())]

drought_df = drought_df[(drought_df['T2M_MAX'] <= drought_df['T2M_MAX'].mean() + 3*drought_df['T2M_MAX'].std()) &
        (drought_df['T2M_MAX'] >= drought_df['T2M_MAX'].mean() - 3*drought_df['T2M_MAX'].std())]

drought_df = drought_df[(drought_df['T2M_MIN'] <= drought_df['T2M_MIN'].mean() + 3*drought_df['T2M_MIN'].std()) &
        (drought_df['T2M_MIN'] >= drought_df['T2M_MIN'].mean() - 3*drought_df['T2M_MIN'].std())]

drought_df = drought_df[(drought_df['T2M_RANGE'] <= drought_df['T2M_RANGE'].mean() + 3*drought_df['T2M_RANGE'].std()) &
        (drought_df['T2M_RANGE'] >= drought_df['T2M_RANGE'].mean() - 3*drought_df['T2M_RANGE'].std())]

drought_df = drought_df[(drought_df['TS'] <= drought_df['TS'].mean() + 3*drought_df['TS'].std()) &
        (drought_df['TS'] >= drought_df['TS'].mean() - 3*drought_df['TS'].std())]

drought_df = drought_df[(drought_df['WS10M'] <= drought_df['WS10M'].mean() + 3*drought_df['WS10M'].std()) &
        (drought_df['WS10M'] >= drought_df['WS10M'].mean() - 3*drought_df['WS10M'].std())]

drought_df = drought_df[(drought_df['WS10M_MAX'] <= drought_df['WS10M_MAX'].mean() + 3*drought_df['WS10M_MAX'].std()) &
        (drought_df['WS10M_MAX'] >= drought_df['WS10M_MAX'].mean() - 3*drought_df['WS10M_MAX'].std())]

drought_df = drought_df[(drought_df['WS10M_MIN'] <= drought_df['WS10M_MIN'].mean() + 3*drought_df['WS10M_MIN'].std()) &
        (drought_df['WS10M_MIN'] >= drought_df['WS10M_MIN'].mean() - 3*drought_df['WS10M_MIN'].std())]

drought_df = drought_df[(drought_df['WS10M_RANGE'] <= drought_df['WS10M_RANGE'].mean() + 3*drought_df['WS10M_RANGE'].std()) &
        (drought_df['WS10M_RANGE'] >= drought_df['WS10M_RANGE'].mean() - 3*drought_df['WS10M_RANGE'].std())]

drought_df = drought_df[(drought_df['WS50M'] <= drought_df['WS50M'].mean() + 3*drought_df['WS50M'].std()) &
        (drought_df['WS50M'] >= drought_df['WS50M'].mean() - 3*drought_df['WS50M'].std())]

drought_df = drought_df[(drought_df['WS50M_MAX'] <= drought_df['WS50M_MAX'].mean() + 3*drought_df['WS50M_MAX'].std()) &
        (drought_df['WS50M_MAX'] >= drought_df['WS50M_MAX'].mean() - 3*drought_df['WS50M_MAX'].std())]

drought_df = drought_df[(drought_df['WS50M_MIN'] <= drought_df['WS50M_MIN'].mean() + 3*drought_df['WS50M_MIN'].std()) &
        (drought_df['WS50M_MIN'] >= drought_df['WS50M_MIN'].mean() - 3*drought_df['WS50M_MIN'].std())]

drought_df = drought_df[(drought_df['WS50M_RANGE'] <= drought_df['WS50M_RANGE'].mean() + 3*drought_df['WS50M_RANGE'].std()) &
        (drought_df['WS50M_RANGE'] >= drought_df['WS50M_RANGE'].mean() - 3*drought_df['WS50M_RANGE'].std())]
print('Total rows = ', len(drought_df.index))
drought_df.head()
drought_df.isnull().sum()

# Univariate Analysis - Distribution of category variables

categorical_column_list = ['score', 'Year', 'Month', 'Day']
drought_df_categorical = drought_df[['score', 'Year', 'Month', 'Day']]

## Distribution of categorical variables

plt.figure(figsize= (10, 40))
for col_name in categorical_column_list:
    plt.figure()
    drought_df_categorical[col_name].value_counts().plot(kind = 'bar')
    x_name = col_name
    plt.xlabel(x_name)
    plt.ylabel('Density')
    plt.title(f'Distribution of {col_name}')
    plt.tight_layout()
    plt.show()


# Bivariate Analysis
plt.scatter(drought_df['Year'], drought_df['score'],c = 'blue')
plt.show()

plt.scatter(drought_df['QV2M'], drought_df['T2M'], C = drought_df['score'])
plt.xlabel('QV2M')
plt.ylabel('T2M')
plt.title('Variation of T2M vs QV2M')
plt.show()

plt.scatter(drought_df['T2M'], drought_df['T2MDEW'], c= drought_df['score'])
plt.xlabel('T2M')
plt.ylabel('T2MDEW')
plt.show()

plt.scatter(drought_df['WS10M'], drought_df['WS50M'], c= drought_df['score'])
plt.xlabel('WS10M')
plt.ylabel('WS50M')
plt.show()

# EXTRACTING DEPENDENT AND INDEPENDENT VARIABLES
independent_variables = drought_df.drop('score', 1)
independent_variables = independent_variables.drop('fips',1)
independent_variables = independent_variables.drop('date', 1)
independent_variables.head()

target = drought_df['score']
target.head()
target.unique()
target.value_counts()

# Correlation between independent variables for Feature Selection

correlation_plot = drought_df_measures.corr()
correlation_plot
correlation_plot.style.background_gradient(cmap = 'RdYlGn')


# Splitting into train and test

x_train, x_test, y_train, y_test = train_test_split(independent_variables, target, test_size = 0.2, random_state = 42)

print('Train features shape', x_train.shape)
print('Train Target shape', y_train.shape)
print('Test featrues shape', x_test.shape)
print('Test Target shape', y_test.shape)

# Standardizing the Data (Data standardization converts data into a standard format that computers can read and understand.)
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
X_train

# Feature Selection using RFE and Random Forest Algorithm

model = RandomForestClassifier(n_estimators = 10) # n_estimators is the hyperparameter, indicating 10 trees
rfe = RFE(model, n_features_to_select = 15) # n_features_to_select is chosen on a trial and error basis
fit = rfe.fit(x_train, y_train)
print('Num Features: %s' % (fit.n_features_))
print('Selected Features: %s' % (fit.support_))
print('Feature Ranking: %s' % (fit.ranking_))
selected_features = independent_variables.columns[(fit.get_support())]
print(selected_features)


independent_variables = independent_variables.drop('PRECTOT', 1)
independent_variables = independent_variables.drop('T2MWET', 1)
independent_variables = independent_variables.drop('WS10M_MAX', 1)
independent_variables = independent_variables.drop('WS10M_MIN', 1)
independent_variables = independent_variables.drop('WS50M_MIN', 1)
independent_variables = independent_variables.drop('Month', 1)
independent_variables.head()

x_train, x_test, y_train, y_test = train_test_split(independent_variables, target, test_size= 0.2, random_state= 42)
print("Train features shape:", x_train.shape)
print('Train target shape: %s' % y_train.shape)
print("Test features shape: {}".format(x_test.shape) )
x_test.shape
print('Test target shape: %s' % y_test.shape)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
x_train


# Fixing Class Imbalance
# Upsampling using SMOTE

sm = SMOTE(random_state = 5)
x_train_ures_SMOTE, y_train_ures_SMOTE = sm.fit_resample(x_train, y_train.ravel())
print('Before Oversampling, the shape of train_x:{}'.format(x_train.shape))