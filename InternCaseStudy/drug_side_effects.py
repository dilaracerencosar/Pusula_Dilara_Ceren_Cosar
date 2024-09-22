import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # You can use 'TkAgg', 'Qt5Agg', or another backend suitable for your environment
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_excel(r"C:\Users\Dilara Ceren Coşar\OneDrive\Masaüstü\Pusula Talent Academy\side_effect_data 1.xlsx")

df.head()
df.info()
df.isnull().sum()
df.describe()

print(df.columns)


# Plot histogram for 'Kilo'
plt.figure(figsize=(12, 6))
sns.histplot(df['Kilo'].dropna(), kde=True)
plt.title('Distribution of Weight (Kilo)')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for 'Boy'
plt.figure(figsize=(12, 6))
sns.histplot(df['Boy'].dropna(), kde=True)
plt.title('Distribution of Height (Boy)')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.show()

print(df.dtypes)

plt.figure(figsize=(12, 6))
sns.countplot(x='Cinsiyet', data=df)
plt.title('Distribution of Gender (Cinsiyet)')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Uyruk', data=df)
plt.title('Distribution of Nationality (Uyruk)')
plt.xlabel('Nationality')
plt.ylabel('Count')
plt.show()


df['Duration'] = (df['Ilac_Bitis_Tarihi'] - df['Ilac_Baslangic_Tarihi']).dt.days
plt.figure(figsize=(12, 6))
sns.histplot(df['Duration'].dropna(), kde=True)
plt.title('Distribution of Medication Duration')
plt.xlabel('Duration (days)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df[['Kilo', 'Boy']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

categorical_features = ['Cinsiyet']  # List of categorical columns to encode
numerical_features = ['Kilo', 'Boy']  # List of numerical columns to keep

X = df[categorical_features + numerical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'  # Keep the numerical features as they are
)

X_encoded = preprocessor.fit_transform(X)

encoded_columns = preprocessor.transformers_[0][1].get_feature_names_out(categorical_features)
final_columns = encoded_columns.tolist() + numerical_features
df_encoded = pd.DataFrame(X_encoded, columns=final_columns)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Kilo', 'Boy']])

df_encoded.to_excel('encoded_data.xlsx', index=False)

