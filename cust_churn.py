# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:39:33 2025

@author: 17086
"""

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Load data
data = pd.read_csv(r"C:\Users\17086\Downloads\Customer Churn.csv")

# ================================================
# Step 2: Data Cleaning

# Check for missing values
print(data.isnull().sum())
data.info()
data.columns
data.describe()

# Handle missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

# Check again for missing values
print(data.isnull().sum())

# Convert binary columns to 1/0
binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

# Convert columns to 'category' dtype for memory efficiency
categorical_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                       'PaperlessBilling', 'Churn', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

for col in categorical_columns:
    data[col] = data[col].astype('category')

# Check for duplicates
print(data.duplicated().sum())
data = data.drop_duplicates()

# Final check on data types and shape
print(data.dtypes)
print(data.shape)

# ================================================
# Step 3: Performing EDA on the Cleaned Data 

# i) Calculate the percentage of customers who have churned
churn_percentage = data['Churn'].value_counts(normalize=True) * 100

# Plot the pie chart
plt.figure(figsize=(4, 4))
churn_percentage.plot.pie(autopct='%1.1f%%', colors=['lightblue', 'salmon'], startangle=90)
plt.title('Percentage of Customers Who Have Churned')
plt.ylabel('')  # Hide the y-label
plt.show()

# ii) Count the number of churn by gender
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=data, hue='Churn', palette='Set2')

# Add title and labels
plt.title('Churn by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')

# Show the plot
plt.show()


# Show the plot
plt.show()

# iii) Analyze churn rates by Senior Citizen 

# CHURN OF CITIZENSHIP WITH PERCENTAGE LABLE
# Ensure the data contains the necessary columns
if 'SeniorCitizen' in data.columns and 'Churn' in data.columns:
    # Create a pivot table to count occurrences of churn by senior citizen status
    churn_by_senior = data.groupby(['SeniorCitizen', 'Churn']).size().unstack()

    # Normalize the data to percentages
    churn_by_senior_percentage = churn_by_senior.div(churn_by_senior.sum(axis=1), axis=0) * 100

    # Plot a stacked bar chart
    ax = churn_by_senior_percentage.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], figsize=(3, 4))

    # Add percentage labels on each segment of the bar
    for i, bar_group in enumerate(ax.containers):
        for bar in bar_group:
            # Get the height of the bar
            height = bar.get_height()
            if height > 0:  # Add label only if height > 0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  
                    bar.get_y() + height / 2,         
                    f'{height:.1f}%',                  
                    ha='center', va='center', fontsize=9, color='black'
                )

    plt.title('Churn by Senior Citizen Status with Percentage Labels', fontsize=14)
    plt.xlabel('Senior Citizen', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(title='Churn', fontsize=10)
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
    plt.tight_layout()

    plt.show()
else:
    print("The dataset must contain 'SeniorCitizen' and 'Churn' columns.")
# ===========================
# iv)# ii) Count the number of customer by seniorcitizen
plt.figure(figsize=(3, 4))
ax =sns.countplot(x='SeniorCitizen', data=data)
ax.bar_lable(ax.containers[0])
plt.title('count of customer by seniorcitizen')

plt.show()


# comparative a greater number of senior citizen have churned
# ===============================
# v) churn by tenure
plt.figure(figsize=(6, 7))

sns.histplot(x = 'tenure', data= data , bins = 70 , hue = 'Churn')
plt.show()
# people who come in the initial month have churnd out frequently
 # ================================
 # vi)# ii) Count of custimers by contract 
# Create the countplot
# Add exact count labels to the bars
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Contract', data=data, hue = "Churn")  # 'tab10' for vibrant colors
ax.bar_label(ax.containers[0], fmt='%d', fontsize=9)  # Add counts on bars
plt.title('Customers by Contract', fontsize=14)
plt.tight_layout()
plt.show()
 # we can conclud that people who contract for monh to month are
 # more likely to churn out in compare to one or two year contract .
#  # an Advice for the Company:
# Focus on promoting longer-term contracts by offering attractive 
# discounts or benefits to encourage loyalty.
# =======================================
#vii)count of multiple columns 
# List of columns for count plots
columns = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies'
]

# Define the number of columns and rows for the subplot grid
n_columns = 3  # Number of columns in the subplot grid
n_rows = (len(columns) + n_columns - 1) // n_columns  # Calculate number of rows needed

# Create the figure and axes for the subplots
fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 4 * n_rows))

# Flatten the axes array for easy iteration (if the grid has multiple rows)
axes = axes.flatten()

# Loop through each column and create the count plot
for i, column in enumerate(columns):
    sns.countplot(x=column, data=data, ax=axes[i],hue = 'Churn')  # Plot each column as a count plot
    axes[i].set_title(f'Count Plot of {column}')  # Set title for each subplot
    axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
    
    # Add count values on top of the bars
    for p in axes[i].patches:
        height = p.get_height()  # Get the height (count) of the bar
        axes[i].text(
            p.get_x() + p.get_width() / 2, height, f'{height}',  # Position of the text
            ha='center', va='bottom', fontsize=10, color='black'  # Text alignment and style
        )

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()

# The mejority of customers who don not churn tend to have services like phone 
# services, internet sevices(particularly DSL) and online services like online 
# backup TechSupport,adn streming TV churn rates are high

# Here are 3 key strategies to reduce churn
 # rate:
# 1) Prioritize value-add services
# 2)Enhance core services
# 3)Improve customer experience
# =================================================
#viii) Count plot for payment method

plt.figure(figsize=(9, 4))
ax=sns.countplot(x='PaymentMethod', data=data, hue = "Churn") 
ax.bar_label(ax.containers[0]) 
plt.title('churn customer by payment_method')
plt.show()

# People who pay thrugh Electronic check have churned high in comoare to others