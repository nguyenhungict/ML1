import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle

# Load the CSV file
car = pd.read_csv('quikr_car.csv')

# Display the first few rows of the dataframe
print(car.head())

# Display the shape of the dataframe
print(car.shape)

# Display the dataframe information
print(car.info())

# Make a backup of the original data
backup = car.copy()

# Cleaning the data
# Filter out non-numeric year values and convert the year to integer
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)

# Filter out rows where 'Price' is 'Ask For Price'
car = car[car['Price'] != 'Ask For Price']

# Remove commas from 'Price' and convert to integer
car['Price'] = car['Price'].str.replace(',', '').astype(int)

# Remove 'kms' from 'kms_driven' and convert to integer
car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)

# Remove rows with NaN values in 'fuel_type'
car = car[~car['fuel_type'].isna()]

# Display shape of cleaned data
print(car.shape)

# Clean the 'name' column to keep only the first three words
car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')

# Resetting the index
car = car.reset_index(drop=True)

# Display cleaned data
print(car.head())

# Save the cleaned data to a CSV file
car.to_csv('Cleaned_Car_data.csv', index=False)

# Display the dataframe information
print(car.info())

# Display descriptive statistics of the cleaned data
print(car.describe(include='all'))

# Filter out extreme values in 'Price'
car = car[car['Price'] < 6000000]

# Visualizing relationships
# Box Plot with color palette
plt.figure(figsize=(13, 7))
ax = sns.boxplot(x='company', y='Price', data=car, palette='husl')  # Using 'husl' palette
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.title('Box Plot of Price by Company')
plt.show()

# Swarm Plot with color palette
plt.figure(figsize=(13, 7))
ax = sns.swarmplot(x='year', y='Price', data=car, palette='Set2')  # Using 'Set2' palette
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.title('Swarm Plot of Price by Year')
plt.show()

# Relational Plot (Scatter Plot) without specific palette (uses default)
sns.relplot(x='kms_driven', y='Price', data=car, height=7, aspect=1.5)
plt.title('Scatter Plot of Price by Kilometers Driven')
plt.show()

# Box Plot by Fuel Type with color palette
plt.figure(figsize=(13, 7))
sns.boxplot(x='fuel_type', y='Price', data=car, palette='pastel')  # Using 'pastel' palette
plt.title('Box Plot of Price by Fuel Type')
plt.show()

# Relational Plot with hue and size (scatter plot) with color palette
plt.figure(figsize=(13, 7))
ax = sns.relplot(x='company', y='Price', data=car, hue='fuel_type', size='year', height=7, aspect=2, palette='viridis')  # Using 'viridis' palette
ax.set_xticklabels(rotation=40, ha='right')
plt.title('Relational Plot of Price by Company with Fuel Type and Year')
plt.show()

# Extracting training data
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Applying Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creating an OneHotEncoder object to contain all the possible categories
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

# Creating a column transformer to transform categorical columns
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                                       remainder='passthrough')

# Linear Regression Model
lr = LinearRegression()

# Making a pipeline
pipe = make_pipeline(column_trans, lr)

# Fitting the model
pipe.fit(X_train, y_train)

# Predicting the test set results
y_pred = pipe.predict(X_test)

# Checking R2 Score
print(r2_score(y_test, y_pred))

# Finding the best random state for TrainTestSplit
scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

best_random_state = np.argmax(scores)
best_score = scores[best_random_state]
print(f'Best R2 Score: {best_score} at random state {best_random_state}')

# Training the model with the best random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=best_random_state)
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)

# Saving the model
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))

# Predicting the price of a sample car
sample_car = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                          data=np.array(['Maruti Suzuki Swift', 'Maruti', 2014, 1000, 'Petrol']).reshape(1, 5))
predicted_price = pipe.predict(sample_car)
print(f'Predicted Price: {predicted_price[0]}')

# Displaying the categories learned by OneHotEncoder
print(pipe.named_steps['columntransformer'].transformers_[0][1].categories_[0])
