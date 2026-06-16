import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop('Cabin', axis=1)

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
plt.savefig('titanic_eda_scatter.png')
plt.title('Fare vs Age')
plt.show()


sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('titanic_eda_heatmap.png')
plt.show()


sns.pairplot(df[['Survived','Age','Fare','Pclass']], hue='Survived')
plt.savefig('titanic_eda_pairplot.png')
plt.show()


