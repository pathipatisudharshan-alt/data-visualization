import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop('Cabin', axis=1)

print("Missing values after basic cleaning:")
print(df.isnull().sum())

sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.savefig('titanic_eda_survival_count.png')
plt.show()

sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival by Gender')
plt.savefig('titanic_eda_count_by_gender.png')
plt.show()

sns.histplot(data=df, x='Age', kde=True)
plt.title('Age Distribution')
plt.savefig('titanic_eda_hist.png')
plt.show()

sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived')
plt.title('Fare vs Age')
plt.savefig('titanic_eda_scatter.png')
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('titanic_eda_heatmap.png')
plt.show()

sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.savefig('titanic_eda_pairplot.png')
plt.show()

print("\nOverall survival rate (%):")
print(df['Survived'].value_counts(normalize=True) * 100)

print("\nSurvival rate by gender (%):")
print(df.groupby('Sex')['Survived'].mean() * 100)

print("\nSurvival rate by class (%):")
print(df.groupby('Pclass')['Survived'].mean() * 100)

def detect_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[col] < lower) | (data[col] > upper)]

fare_outliers = detect_outliers_iqr(df, 'Fare')
age_outliers = detect_outliers_iqr(df, 'Age')
print(f"\nFare outliers: {len(fare_outliers)}")
print(f"Age outliers: {len(age_outliers)}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, y='Fare')
plt.title('Fare Outliers')
plt.subplot(1, 2, 2)
sns.boxplot(data=df, y='Age')
plt.title('Age Outliers')
plt.tight_layout()
plt.savefig('titanic_eda_outliers.png')
plt.show()

df['Fare_capped'] = df['Fare'].clip(upper=df['Fare'].quantile(0.99))

df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.')
df['Title'] = df['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
     'Sir', 'Jonkheer', 'Dona'], 'Rare'
)
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

sns.barplot(data=df, x='Title', y='Survived')
plt.xticks(rotation=45)
plt.title('Survival by Title')
plt.tight_layout()
plt.savefig('titanic_eda_title.png')
plt.show()

sns.barplot(data=df, x='FamilySize', y='Survived')
plt.title('Survival by Family Size')
plt.savefig('titanic_eda_familysize.png')
plt.show()

df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])

print("\nFinal cleaned + engineered dataframe preview:")
print(df.head())

print("\nOne-hot encoded dataframe shape:", df_encoded.shape)

df.to_csv('titanic_cleaned_engineered.csv', index=False)
df_encoded.to_csv('titanic_encoded.csv', index=False)

print("\nDone. Saved: titanic_cleaned_engineered.csv, titanic_encoded.csv")
