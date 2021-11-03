# %% [markdown]
# # Import Modules

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# %%
file_path = 'train.csv'

data = pd.read_csv(file_path, index_col=0)

# %% [markdown]
# # 1 Exploratory Data Analysis
#
# ## 1.1 Preliminary observations

# %%
data.head()


# %%
data.tail()


# %%
data.shape

# %% [markdown]
# ### 1.1.1 Numerical columns

# %%
num_cols = data.select_dtypes(exclude=['object']).columns

num_cols


# %%
len(num_cols)


# %%
data.select_dtypes(exclude=['object']).describe().round(decimals=2)

# %% [markdown]
# ### 1.1.2 Categorical columns

# %%
cat_cols = data.select_dtypes(include=['object']).columns

cat_cols


# %%
len(cat_cols)


# %%
data.select_dtypes(include=['object']).describe()

# %% [markdown]
# ## 1.2 Exploring numerical columns
#
# ### 1.2.1 Skew of target column

# %%
target = data.SalePrice

plt.figure()
sns.distplot(target)
plt.title('Distribution of SalePrice')
plt.show()


# %%
sns.distplot(np.log(target))
plt.title('Distribution of log-transformed SalePrice')
plt.xlabel('log(SalePrice)')
plt.show()


# %%
print('SalePrice has a skew of ' + str(target.skew().round(decimals=2)) +
      ' while the log-transformed SalePrice improves the skew to ' + str(np.log(target).skew().round(decimals=2)))

# %% [markdown]
# ### 1.2.2 Distribution of attributes

# %%
num_attrs = data.select_dtypes(
    exclude=['object']).drop('SalePrice', axis=1).copy()

fig = plt.figure(figsize=(12, 18))
for i in range(len(num_attrs.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.distplot(num_attrs.iloc[:, i].dropna())
    plt.xlabel(num_attrs.columns[i])

plt.tight_layout()
plt.show()

# %% [markdown]
# Skewed dists could be potentially be log-transformed:
# + LotFrontage
# + LotArea
# + 1stFlrSF
# + GrLivArea
# + OpenPorchSF
# %% [markdown]
# ### 1.2.3 Finding outliers
# %% [markdown]
# Univariate analysis

# %%
fig = plt.figure(figsize=(12, 18))

for i in range(len(num_attrs.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.boxplot(y=num_attrs.iloc[:, i])

plt.tight_layout()
plt.show()

# %% [markdown]
# Bivariate analysis

# %%
fig = plt.figure(figsize=(20, 20))

for i in range(len(num_attrs.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.scatterplot(num_attrs.iloc[:, i], target)

plt.tight_layout()
plt.show()

# %% [markdown]
# Correlation

# %%
correlation = data.corr()

f, ax = plt.subplots(figsize=(14, 12))
plt.title('Correlation of numerical attributes', size=16)
sns.heatmap(correlation)
plt.show()


# %%
correlation['SalePrice'].sort_values(ascending=False).head(15)


# %%
corr_to_price = correlation['SalePrice']
n_cols, n_rows = 5, 8
fix, ax_arr = plt.subplots(n_rows, n_cols, figsize=(16, 20), sharey=True)
plt.subplots_adjust(bottom=-0.8)
for j in range(n_rows):
    for i in range(n_cols):
        plt.sca(ax_arr[j, i])
        index = i + j*n_cols
        if index < len(num_cols):
            plt.scatter(data[num_cols[index]], data.SalePrice)
            plt.xlabel(num_cols[index])
            plt.title('Corr to SalePrice = ' +
                      str(np.around(corr_to_price[index], decimals=3)))
plt.show()

# %% [markdown]
# Missing/Null values in numerical cols

# %%
num_attrs.isna().sum().sort_values(ascending=False).head()

# %% [markdown]
# ## 1.3 Exploring categorical columns

# %%
cat_cols


# %%
var = data['KitchenQual']
f, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(var, data.SalePrice)
plt.show()


# %%
f, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data['Neighborhood'], data['SalePrice'])
plt.xticks(rotation=40)
plt.show()


# %%
# Count of categories within Neighborhood attribute
fig = plt.figure(figsize=(12.5, 4))
sns.countplot(x='Neighborhood', data=data)
plt.xticks(rotation=90)
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# Missing/null values in categorical columns

# %%
data[cat_cols].isna().sum().sort_values(ascending=False).head(17)

# %% [markdown]
# # 2. Data Cleaning & Preprocessing
#
# ## 2.1 Dealing w/ missing/null values

# %%
# Create copy of dataset
data_copy = data.copy()

# Numerical columns
data_copy['MasVnrArea'] = data_copy['MasVnrArea'].fillna(0)

# Categorical columns
cat_cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                      'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',
                      'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',
                      'MasVnrType']

for cat in cat_cols_fill_none:
    data_copy[cat] = data_copy[cat].fillna('None')


# %%
# Check for outstanding missing/null values
# Use imputer for these
data_copy.isna().sum().sort_values(ascending=False).head()

# %% [markdown]
# ## 2.2 Adressing outliers

# %%
# Remove outliers based on observations on scatter plots against SalePrice

# %% [markdown]
# ## 2.3 Transforming data to reduce skew

# %%
data_copy['SalePrice'] = np.log(data_copy['SalePrice'])
data_copy = data_copy.rename(columns={'SalePrice': 'SalePrice_log'})

# %% [markdown]
# # 3 Feature Selection & Engineering
#
# Considering highly-correlated features

# %%
transformed_corr = data_copy.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(transformed_corr)

# %% [markdown]
# Highly-correlated attrubtes include
#
# - GarageCars and GarageArea
# - YearBuilt and GarageYrBlt
# - GrLiveArea and TotRmsAbvGrd
# - TotalBsmtSF and 1stFlrSF
# %% [markdown]
# Perform feature selection and encoding of categorical columns

# %%
# Remove attrs that were identified for excluding when viewing scatter plots & corr values
attrs_drop = ['SalePrice_log', 'MiscVal', 'MSSubClass', 'MoSold', 'YrSold',
              'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd']

X = data_copy.drop(attrs_drop, axis=1)

y = data_copy.SalePrice_log

X = pd.get_dummies(X)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Normalization
normalizer = StandardScaler()
train_X = normalizer.fit_transform(train_X)
val_X = normalizer.transform(val_X)

# Final imputation of missing data
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
val_X = my_imputer.transform(val_X)

# %% [markdown]
# # ML Algorithms

# %%


# %% [markdown]
# Inverser trafo from log(SalePrice) to SalePrice

# %%

def inverse_y(transformed_y):
    return np.exp(transformed_y)


# %%
# Series to collate mae for each algo
mea_compare = pd.Series()
mea_compare.index.name = 'Algorithm'

# %% [markdown]
# ## Decision Tree

# %%
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(train_X, train_y)
dt_pred = dt_model.predict(val_X)
dt_val_mae = mean_absolute_error(inverse_y(val_y), inverse_y(dt_pred))
print(f'Validation MAE for DecisionTree: {dt_val_mae:.2f}')
mea_compare['DecisionTree'] = dt_val_mae


# %%
# w/ max leaf nodes of 90
dt_model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=90)
dt_model.fit(train_X, train_y)
dt_pred = dt_model.predict(val_X)
dt_val_mae = mean_absolute_error(inverse_y(val_y), inverse_y(dt_pred))
print(
    f'Validation MAE for DecisionTree wit max leaf nodes of 90: {dt_val_mae:.2f}')
mea_compare['DecisionTree_with_max_leaf_nodes'] = dt_val_mae

# %% [markdown]
# ## Random Forest

# %%
rf_model = RandomForestRegressor(random_state=5)
rf_model.fit(train_X, train_y)
rf_pred = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(inverse_y(val_y), inverse_y(rf_pred))
print(f'Validation MAE for Random Forest Model: {rf_val_mae:.2f}')
mea_compare['RandomForest'] = rf_val_mae

# %% [markdown]
# ## XGBoost

# %%
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb_model.fit(train_X, train_y, early_stopping_rounds=5,
              eval_set=[(val_X, val_y)], verbose=True)
xgb_pred = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(inverse_y(val_y), inverse_y(xgb_pred))
print(f'Validation MAE for XGBoost Model: {xgb_val_mae:.2f}')
mea_compare['XGBoost'] = xgb_val_mae

# %% [markdown]
# ## Linear Regression

# %%
# linear_model = LinearRegression()
# linear_model.fit(train_X, train_y)
# linear_pred = linear_model.predict(val_X)
# linear_val_mae = mean_absolute_error(inverse_y(val_y), inverse_y(linear_pred))
# print(f'Validation MAE for Linear Model: {linear_val_mae:.2f}') # dtype('float64') problem
# mea_compare['LinearRegression'] = linear_val_mae

# %% [markdown]
# ## Lasso

# %%
lasso_model = Lasso(alpha=0.0005, random_state=5)
lasso_model.fit(train_X, train_y)
lasso_pred = lasso_model.predict(val_X)
lasso_val_mae = mean_absolute_error(inverse_y(val_y), inverse_y(lasso_pred))
print(f'Validation MAE for Lasso Model: {lasso_val_mae:.2f}')
mea_compare['Lasso'] = lasso_val_mae

# %% [markdown]
# ## Ridge

# %%
ridge_model = Ridge(alpha=0.0002, random_state=5)
ridge_model.fit(train_X, train_y)
ridge_pred = ridge_model.predict(val_X)
ridge_val_mae = mean_absolute_error(inverse_y(val_y), inverse_y(ridge_pred))
print(f'Validation MAE for Ridge Model: {ridge_val_mae:.2f}')
mea_compare['Ridge'] = ridge_val_mae

# %% [markdown]
# ## ElasticNet

# %%
elastic_net_model = ElasticNet(alpha=0.02, random_state=5, l1_ratio=0.7)
elastic_net_model.fit(train_X, train_y)
elastic_net_pred = elastic_net_model.predict(val_X)
elastic_net_val_mae = mean_absolute_error(
    inverse_y(val_y), inverse_y(elastic_net_pred))
print(f'Validation MAE for ElasticNet Model: {elastic_net_val_mae:.2f}')
mea_compare['ElasticNet'] = elastic_net_val_mae


# %%
mea_compare


# %%
mea_compare

# %% [markdown]
# ### Cross Validation

# %%


# %%
imputer = SimpleImputer()
imputed_X = imputer.fit_transform(X)
n_folds = 10


# %%
scores = cross_val_score(lasso_model, imputed_X, y,
                         scoring='neg_mean_squared_error', cv=n_folds)
lasso_mae_scores = np.sqrt(-scores)

print('For Lasso model: ')
print(lasso_mae_scores.round(decimals=2))
print(f'Mean RMSE = {lasso_mae_scores.mean().round(decimals=3)}')
print(f'Error std deviation = {lasso_mae_scores.std().round(decimals=3)}')

# %% [markdown]
# # ML Algorithm Selection

# %%
# Grid search for hyperparameter tuning

# Tuning Lasso
param_grid = [{'alpha': [0.0007, 0.0005, 0.005]}]
top_reg = Lasso()

grid_search = GridSearchCV(
    top_reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(imputed_X, y)
grid_search.best_params_


# %%
# path to file for predictions
test_data_path = 'test.csv'

# read test data
test_data = pd.read_csv(test_data_path)


# %%
test_X = test_data.copy()

test_X['MasVnrArea'] = test_X['MasVnrArea'].fillna(0)

for cat in cat_cols_fill_none:
    test_X[cat] = test_X[cat].fillna('None')

if 'SalePrice_log' in attrs_drop:
    attrs_drop.remove('SalePrice_log')

test_X = test_X.drop(attrs_drop, axis=1)

test_X = pd.get_dummies(test_X)

final_train, final_test = X.align(test_X, join='left', axis=1)

final_test_imputed = my_imputer.transform(final_test)


# %%
# Create final model

final_model = Lasso(alpha=0.0005, random_state=5)
final_train_imputed = my_imputer.fit_transform(final_train)

final_model.fit(final_train_imputed, y)


# %%
# Make predictions for submission

test_preds = final_model.predict(final_test_imputed)

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': inverse_y(test_preds)})

output.to_csv('submission.csv', index=False)


# %%
