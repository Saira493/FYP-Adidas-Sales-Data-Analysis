#!/usr/bin/env python
# coding: utf-8

# # Dissertation Code

# ## Libraries

# In[ ]:


from google.colab import drive
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ## Google Drive Mounting

# In[ ]:


drive.mount('/content/drive')


# ## File Import

# In[ ]:


file_path = '/content/drive/MyDrive/Dissertation/Adidas US Sales Datasets 16.csv'
df = pd.read_csv(file_path)

print(df.head())
print(df.info())


# ## Data Preprocessing

# In[ ]:


df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])

def clean_monetary_column(column):
    return pd.to_numeric(column.replace({'\$': '', ',': '', '%': ''}, regex=True), errors='coerce')

df['Price per Unit'] = clean_monetary_column(df['Price per Unit'])
df['Total Sales'] = clean_monetary_column(df['Total Sales'])
df['Operating Profit'] = clean_monetary_column(df['Operating Profit'])
df['Operating Margin'] = clean_monetary_column(df['Operating Margin'])

df['Units Sold'] = pd.to_numeric(df['Units Sold'].str.replace(',', ''), errors='coerce')

df['Total Sales'] = df['Price per Unit'] * df['Units Sold']
df['Operating Profit'] = df['Total Sales'] * (df['Operating Margin'] / 100)

print(df.head())


# ## Time Series Analysis

# In[ ]:


df['Total Sales'] = pd.to_numeric(df['Total Sales'], errors='coerce')

df.set_index('Invoice Date', inplace=True)

monthly_sales = df['Total Sales'].resample('M').sum()

plt.figure(figsize=(14, 7))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o', color='blue')
plt.title('Monthly Sales Trends for Adidas US', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


decomposition = seasonal_decompose(monthly_sales, model='additive')
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.show()


# In[ ]:


quarterly_sales = df['Total Sales'].resample('Q').sum()

plt.figure(figsize=(14, 7))
sns.lineplot(x=quarterly_sales.index, y=quarterly_sales.values, marker='o', color='green')
plt.title('Quarterly Sales Trends for Adidas US', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


decomposition = seasonal_decompose(quarterly_sales, model='additive')
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.show()


# ## Total Sales by Region

# In[ ]:


df['Total Sales'] = pd.to_numeric(df['Total Sales'], errors='coerce')

region_sales = df.groupby('Region')['Total Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=region_sales.values, y=region_sales.index, palette='viridis')
plt.title('Total Sales by Region for Adidas US', fontsize=16)
plt.xlabel('Total Sales', fontsize=14)
plt.ylabel('Region', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Top Product Performance by Sales

# In[ ]:


product_sales = df.groupby('Product')['Total Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(x=product_sales.values, y=product_sales.index, palette='viridis')
plt.title(f'Products Performance by Total Sales', fontsize=16)
plt.xlabel('Total Sales', fontsize=14)
plt.ylabel('Product', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Top Product Elasticity

# In[ ]:


df['Price Change (%)'] = df.groupby('Product')['Price per Unit'].pct_change() * 100
df['Units Sold Change (%)'] = df.groupby('Product')['Units Sold'].pct_change() * 100

df = df[df['Price Change (%)'] != 0]

df['Price Elasticity'] = df['Units Sold Change (%)'] / df['Price Change (%)']

df = df.dropna(subset=['Price Elasticity'])

df = df[~df['Price Elasticity'].isin([float('inf'), float('-inf')])]

product_elasticity = df.groupby('Product')['Price Elasticity'].mean().sort_values(ascending=False)

print("Average Price Elasticity by Product:")
print(product_elasticity.head())

top_n = 10
top_elasticity_products = product_elasticity.head(top_n)

print(f"Top {top_n} Products by Price Elasticity:")
print(top_elasticity_products)

plt.figure(figsize=(14, 8))
sns.barplot(x=top_elasticity_products.values, y=top_elasticity_products.index, palette='plasma')
plt.title(f'Top {top_n} Products by Price Elasticity for Adidas US', fontsize=16)
plt.xlabel('Average Price Elasticity', fontsize=14)
plt.ylabel('Product', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Retailer Performance Analysis

# In[ ]:


# Group by 'Retailer' and aggregate performance metrics
retailer_performance = df.groupby('Retailer').agg(
    total_sales=('Total Sales', 'sum'),
    total_units_sold=('Units Sold', 'sum'),
    avg_price_per_unit=('Price per Unit', 'mean'),
    total_operating_profit=('Operating Profit', 'sum')
).reset_index()

# Sort retailers by total sales
retailer_performance_sorted = retailer_performance.sort_values(by='total_sales', ascending=False)

# Print to debug
print("Retailer Performance:")
print(retailer_performance_sorted.head())

# Plot total sales by retailer
plt.figure(figsize=(14, 8))
sns.barplot(x='total_sales', y='Retailer', data=retailer_performance_sorted, palette='viridis')
plt.title('Total Sales by Retailer', fontsize=16)
plt.xlabel('Total Sales ($)', fontsize=14)
plt.ylabel('Retailer', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot total units sold by retailer
plt.figure(figsize=(14, 8))
sns.barplot(x='total_units_sold', y='Retailer', data=retailer_performance_sorted, palette='viridis')
plt.title('Total Units Sold by Retailer', fontsize=16)
plt.xlabel('Total Units Sold', fontsize=14)
plt.ylabel('Retailer', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot average price per unit by retailer
plt.figure(figsize=(14, 8))
sns.barplot(x='avg_price_per_unit', y='Retailer', data=retailer_performance_sorted, palette='viridis')
plt.title('Average Price per Unit by Retailer', fontsize=16)
plt.xlabel('Average Price per Unit ($)', fontsize=14)
plt.ylabel('Retailer', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot total operating profit by retailer
plt.figure(figsize=(14, 8))
sns.barplot(x='total_operating_profit', y='Retailer', data=retailer_performance_sorted, palette='viridis')
plt.title('Total Operating Profit by Retailer', fontsize=16)
plt.xlabel('Total Operating Profit ($)', fontsize=14)
plt.ylabel('Retailer', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Sales Method Analysis

# In[ ]:


# Group by 'Sales Method' and aggregate performance metrics
sales_method_performance = df.groupby('Sales Method').agg(
    total_sales=('Total Sales', 'sum'),
    total_units_sold=('Units Sold', 'sum'),
    avg_price_per_unit=('Price per Unit', 'mean'),
    total_operating_profit=('Operating Profit', 'sum')
).reset_index()

# Sort sales methods by total sales
sales_method_performance_sorted = sales_method_performance.sort_values(by='total_sales', ascending=False)

print("Sales Method Performance:")
print(sales_method_performance_sorted)

# Plot total sales by sales method
plt.figure(figsize=(14, 8))
sns.barplot(x='total_sales', y='Sales Method', data=sales_method_performance_sorted, palette='coolwarm')
plt.title('Total Sales by Sales Method', fontsize=16)
plt.xlabel('Total Sales ($)', fontsize=14)
plt.ylabel('Sales Method', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot total units sold by sales method
plt.figure(figsize=(14, 8))
sns.barplot(x='total_units_sold', y='Sales Method', data=sales_method_performance_sorted, palette='coolwarm')
plt.title('Total Units Sold by Sales Method', fontsize=16)
plt.xlabel('Total Units Sold', fontsize=14)
plt.ylabel('Sales Method', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot average price per unit by sales method
plt.figure(figsize=(14, 8))
sns.barplot(x='avg_price_per_unit', y='Sales Method', data=sales_method_performance_sorted, palette='coolwarm')
plt.title('Average Price per Unit by Sales Method', fontsize=16)
plt.xlabel('Average Price per Unit ($)', fontsize=14)
plt.ylabel('Sales Method', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot total operating profit by sales method
plt.figure(figsize=(14, 8))
sns.barplot(x='total_operating_profit', y='Sales Method', data=sales_method_performance_sorted, palette='coolwarm')
plt.title('Total Operating Profit by Sales Method', fontsize=16)
plt.xlabel('Total Operating Profit ($)', fontsize=14)
plt.ylabel('Sales Method', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Operating Margin Analysis

# In[ ]:


# Calculate Operating Margin
df['Operating Margin'] = (df['Operating Profit'] / df['Total Sales']) * 100

df = df.dropna(subset=['Operating Margin'])

# Group by 'Retailer'
operating_margin_retailer = df.groupby('Retailer')['Operating Margin'].mean().reset_index()
operating_margin_retailer = operating_margin_retailer.sort_values(by='Operating Margin', ascending=False)

# Group by 'Region'
operating_margin_region = df.groupby('Region')['Operating Margin'].mean().reset_index()
operating_margin_region = operating_margin_region.sort_values(by='Operating Margin', ascending=False)

# Group by 'Sales Method'
operating_margin_sales_method = df.groupby('Sales Method')['Operating Margin'].mean().reset_index()
operating_margin_sales_method = operating_margin_sales_method.sort_values(by='Operating Margin', ascending=False)

print("Operating Margin by Retailer:")
print(operating_margin_retailer)

print("Operating Margin by Region:")
print(operating_margin_region)

print("Operating Margin by Sales Method:")
print(operating_margin_sales_method)

# Plot operating margin by retailer
plt.figure(figsize=(14, 8))
sns.barplot(x='Operating Margin', y='Retailer', data=operating_margin_retailer, palette='crest')
plt.title('Operating Margin by Retailer', fontsize=16)
plt.xlabel('Operating Margin (%)', fontsize=14)
plt.ylabel('Retailer', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot operating margin by region
plt.figure(figsize=(14, 8))
sns.barplot(x='Operating Margin', y='Region', data=operating_margin_region, palette='crest')
plt.title('Operating Margin by Region', fontsize=16)
plt.xlabel('Operating Margin (%)', fontsize=14)
plt.ylabel('Region', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot operating margin by sales method
plt.figure(figsize=(14, 8))
sns.barplot(x='Operating Margin', y='Sales Method', data=operating_margin_sales_method, palette='crest')
plt.title('Operating Margin by Sales Method', fontsize=16)
plt.xlabel('Operating Margin (%)', fontsize=14)
plt.ylabel('Sales Method', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Cost Analysis

# In[ ]:


# Calculate Cost
df['Cost'] = df['Total Sales'] - df['Operating Profit']

# Calculate Cost per Unit Sold
df['Cost per Unit'] = df['Cost'] / df['Units Sold']

# Group by 'Retailer'
cost_per_retailer = df.groupby('Retailer').agg(
    total_cost=('Cost', 'sum'),
    total_units_sold=('Units Sold', 'sum'),
    avg_cost_per_unit=('Cost per Unit', 'mean'),
    total_operating_profit=('Operating Profit', 'sum')
).reset_index()

# Calculate the ratio of operating profit to total cost
cost_per_retailer['profit_to_cost_ratio'] = cost_per_retailer['total_operating_profit'] / cost_per_retailer['total_cost']

# Sort by profit-to-cost ratio
cost_per_retailer_sorted = cost_per_retailer.sort_values(by='profit_to_cost_ratio', ascending=False)

print("Cost and Profit Analysis by Retailer:")
print(cost_per_retailer_sorted.head())

# Plot total cost by retailer
plt.figure(figsize=(14, 8))
sns.barplot(x='total_cost', y='Retailer', data=cost_per_retailer_sorted, palette='viridis')
plt.title('Total Cost by Retailer', fontsize=16)
plt.xlabel('Total Cost ($)', fontsize=14)
plt.ylabel('Retailer', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot average cost per unit by retailer
plt.figure(figsize=(14, 8))
sns.barplot(x='avg_cost_per_unit', y='Retailer', data=cost_per_retailer_sorted, palette='viridis')
plt.title('Average Cost per Unit by Retailer', fontsize=16)
plt.xlabel('Average Cost per Unit ($)', fontsize=14)
plt.ylabel('Retailer', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot profit-to-cost ratio by retailer
plt.figure(figsize=(14, 8))
sns.barplot(x='profit_to_cost_ratio', y='Retailer', data=cost_per_retailer_sorted, palette='viridis')
plt.title('Profit-to-Cost Ratio by Retailer', fontsize=16)
plt.xlabel('Profit-to-Cost Ratio', fontsize=14)
plt.ylabel('Retailer', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Seasonal Effect

# In[ ]:


# Extract month and quarter
df['Month'] = df['Invoice Date'].dt.month
df['Quarter'] = df['Invoice Date'].dt.to_period('Q')

# Aggregate data by month
monthly_sales = df.groupby('Month')['Total Sales'].sum().reset_index()
monthly_sales['Month Name'] = monthly_sales['Month'].apply(lambda x: pd.to_datetime(f'2020-{x}-01').strftime('%B'))
monthly_sales = monthly_sales.sort_values(by='Month')

# Aggregate data by quarter
quarterly_sales = df.groupby('Quarter')['Total Sales'].sum().reset_index()
quarterly_sales['Quarter'] = quarterly_sales['Quarter'].apply(lambda x: f'Q{x.quarter} {x.year}')

print("Monthly Sales Data:")
print(monthly_sales)

print("Quarterly Sales Data:")
print(quarterly_sales)

# Plot monthly sales
plt.figure(figsize=(14, 8))
sns.lineplot(x='Month Name', y='Total Sales', data=monthly_sales, marker='o', color='blue')
plt.title('Monthly Sales Trends', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Total Sales ($)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot quarterly sales
plt.figure(figsize=(14, 8))
sns.barplot(x='Quarter', y='Total Sales', data=quarterly_sales, palette='coolwarm')
plt.title('Quarterly Sales Trends', fontsize=16)
plt.xlabel('Quarter', fontsize=14)
plt.ylabel('Total Sales ($)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Sales Method Effectiveness

# In[ ]:


sales_method_analysis = df.groupby('Sales Method').agg(
    Total_Sales=('Total Sales', 'sum'),
    Average_Sales=('Total Sales', 'mean'),
    Total_Units_Sold=('Units Sold', 'sum'),
    Average_Units_Sold=('Units Sold', 'mean'),
    Total_Operating_Profit=('Operating Profit', 'sum'),
    Average_Operating_Profit=('Operating Profit', 'mean')
).reset_index()

print("Sales Method Effectiveness Analysis:")
print(sales_method_analysis)

# Plot Total Sales by Sales Method
plt.figure(figsize=(10, 6))
sns.barplot(x='Sales Method', y='Total_Sales', data=sales_method_analysis, palette='viridis')
plt.title('Total Sales by Sales Method', fontsize=16)
plt.xlabel('Sales Method', fontsize=14)
plt.ylabel('Total Sales ($)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot Total Units Sold by Sales Method
plt.figure(figsize=(10, 6))
sns.barplot(x='Sales Method', y='Total_Units_Sold', data=sales_method_analysis, palette='plasma')
plt.title('Total Units Sold by Sales Method', fontsize=16)
plt.xlabel('Sales Method', fontsize=14)
plt.ylabel('Total Units Sold', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot Average Operating Profit by Sales Method
plt.figure(figsize=(10, 6))
sns.barplot(x='Sales Method', y='Average_Operating_Profit', data=sales_method_analysis, palette='coolwarm')
plt.title('Average Operating Profit by Sales Method', fontsize=16)
plt.xlabel('Sales Method', fontsize=14)
plt.ylabel('Average Operating Profit ($)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Region State City Analysis

# In[ ]:


region_state_city_analysis = df.groupby(['Region', 'State', 'City']).agg(
    Total_Sales=('Total Sales', 'sum'),
    Total_Units_Sold=('Units Sold', 'sum'),
    Total_Operating_Profit=('Operating Profit', 'sum')
).reset_index()

print("Region-State-City Triad Analysis:")
print(region_state_city_analysis)

# Plot Total Sales by Region
plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Total_Sales', data=region_state_city_analysis, palette='viridis', ci=None)
plt.title('Total Sales by Region', fontsize=16)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Total Sales ($)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot Total Sales by State
plt.figure(figsize=(14, 8))
state_sales = region_state_city_analysis.groupby('State')['Total_Sales'].sum().reset_index().sort_values(by='Total_Sales', ascending=False)
sns.barplot(x='Total_Sales', y='State', data=state_sales, palette='plasma')
plt.title('Total Sales by State', fontsize=16)
plt.xlabel('Total Sales ($)', fontsize=14)
plt.ylabel('State', fontsize=14)
plt.tight_layout()
plt.show()

# Plot Total Sales by City
plt.figure(figsize=(16, 10))
city_sales = region_state_city_analysis.groupby('City')['Total_Sales'].sum().reset_index().sort_values(by='Total_Sales', ascending=False)
sns.barplot(x='Total_Sales', y='City', data=city_sales.head(20), palette='coolwarm')
plt.title('Top 20 Cities by Total Sales', fontsize=16)
plt.xlabel('Total Sales ($)', fontsize=14)
plt.ylabel('City', fontsize=14)
plt.tight_layout()
plt.show()


# ## Feature Selection

# In[ ]:


# Encode categorical variables
label_encoders = {}
for column in ['Retailer', 'Region', 'State', 'City', 'Product', 'Sales Method']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

df = df.dropna(subset=['Total Sales', 'Units Sold', 'Operating Profit', 'Price per Unit'])

# Feature Selection using RandomForest and SelectKBest

# Defining the features and target variable
features = df.drop(columns=['Total Sales', 'Invoice Date'])
target = df['Total Sales']

# Feature importance using Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(features, target)
feature_importances = pd.Series(rf.feature_importances_, index=features.columns)
feature_importances = feature_importances.sort_values(ascending=False)

print("Feature importances using RandomForestRegressor:")
print(feature_importances)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.title('Feature Importances using RandomForestRegressor', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()

# Feature selection using SelectKBest
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(features, target)
feature_scores = pd.Series(selector.scores_, index=features.columns)
feature_scores = feature_scores.sort_values(ascending=False)

print("\nFeature scores using SelectKBest with f_regression:")
print(feature_scores)

# Plot the feature scores
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_scores.values, y=feature_scores.index, palette='plasma')
plt.title('Feature Scores using SelectKBest (f_regression)', fontsize=16)
plt.xlabel('Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()


# In[ ]:


# Extract feature importances from the best model
feature_importances = best_rf.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from Random Forest Model')
plt.gca().invert_yaxis()
plt.show()


# ## Model Selection

# In[ ]:


# Preparing the data based on the most important features
selected_features = ['Operating Profit', 'Units Sold', 'Operating Margin']
X = df[selected_features]
y = df['Total Sales']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate the models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'Mean Squared Error': mse, 'R2 Score': r2}

# Display the results
print("Model Performance:")
for model_name, metrics in results.items():
    print(f"{model_name}: MSE = {metrics['Mean Squared Error']:.2f}, R2 Score = {metrics['R2 Score']:.2f}")


# ## Random Forest Regressor training

# In[ ]:


# Adjusted parameter grid without 'auto'
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],  # Number of trees
    'max_features': ['sqrt', 'log2', None],     # Number of features to consider at every split
    'max_depth': [None, 10, 20, 30, 40, 50],    # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],            # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],              # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]                  # Method of selecting samples for training each tree
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings that are sampled
    cv=5,  # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit RandomizedSearchCV to the data
random_search.fit(X_train, y_train)

# Best parameters from RandomizedSearchCV
best_params = random_search.best_params_
print("Best parameters found: ", best_params)

# Evaluate the best model on the test set
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Best Model Performance: MSE = {mse:.2f}, R2 Score = {r2:.2f}")


# In[ ]:


# Predict on the test set
y_pred = best_rf.predict(X_test)

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()


# ## Cross Validation

# In[ ]:


# Cross-validate the best model
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='r2')
print(f"Cross-Validation R2 Scores: {cv_scores}")
print(f"Average Cross-Validation R2 Score: {np.mean(cv_scores)}")


# ## Arima Training

# In[ ]:


train_data = df['Total Sales'][:20]

model = ARIMA(train_data, order=(3,1,1))
model_fit = model.fit()

# Forecast the next 3 months
forecast = model_fit.forecast(steps=3)

print("Forecasted values for the next 3 months:", forecast)

# Plot the original data and the forecasted data
plt.figure(figsize=(12, 6))
plt.plot(df['YearMonth'], df['Total Sales'], marker='o', label='Original Data')
plt.plot(df['YearMonth'][21:], forecast, marker='x', linestyle='--', color='red', label='Forecasted Data')

plt.title('Total Sales Over Time with Forecast')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.xticks(df['YearMonth'],rotation=45)
plt.legend()
plt.tight_layout()

plt.show()

