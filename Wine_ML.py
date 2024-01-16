import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import statsmodels.api as sm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
df1 = pd.read_csv('wine.csv')
df = pd.read_csv('wine.csv')
# First, Let's scale the data, as the mertics are all over the place. 
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

# Explore the data through groupby functions
# the goal is to see if there are any clear stand out columns
melted = pd.melt(
    scaled_df, id_vars='quality', var_name='Column', value_name='Value')
means = melted.groupby(
    ['quality', 'Column']).mean().reset_index()
# Plotting using Seaborn as a line graph
plt.figure(figsize=(8, 6))
sns.lineplot(x='quality', y='Value', hue='Column', data=melted)
plt.title('Mean values of columns by Quality')
plt.xticks(rotation=45)
plt.xlabel('Quality')
plt.ylabel('Mean Value')
plt.legend(title='Column', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# let's create a dummy variable to keep this binary
# I'm only interested in drinking wines above a 7. 
df['quality'] = df['quality'].astype(float)
df['dummy'] = np.where(df['quality'] >= 7, '1', '0')
# okay people like higher alcohol and higher PH, the rest are pretty middle of the road
# So lets build a training and test dataset
# Splitting the data into training and testing sets (70% train, 30% test)
# Assuming 'quality' as the outcome variable



df['pH'] = pd.to_numeric(df['pH'], errors='coerce')
df['alcohol'] = pd.to_numeric(df['alcohol'], errors='coerce')
df['dummy'] = pd.to_numeric(df['dummy'], errors='coerce')
df = df.dropna()  # Drop rows with NaN values, if necessary
X = df[['pH', 'alcohol']]  # Independent variables
y = df['dummy']  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))






# Assuming X_train is a pandas DataFrame
x_min = X_train.iloc[:, 0].min() - 1
x_max = X_train.iloc[:, 0].max() + 1
y_min = X_train.iloc[:, 1].min() - 1
y_max = X_train.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape) 
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
class_0 = X_train[y_train == 0] 
class_1 = X_train[y_train == 1]
class_2 = X_train[y_train == 2]
# Plotting with the NumPy array
plt.scatter(class_0[:, 0], class_0[:, 1], color='darkgreen', label='< 7 Quality Score')
plt.scatter(class_1[:, 0], class_1[:, 1], color='tan', label='>= 7 Quality Score')
plt.xlabel('pH')
plt.ylabel('Alcohol')
plt.title('Predicting the Quality of Wine by pH and Alcohol Percentage')
plt.legend()
plt.show()




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the model
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('GNB')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Logisticregression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Logistic')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#random forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print('Random forst')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

df = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred})


# Create an empty list to store dummy variable values
dummy_variable = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    true_label = row['y_test']
    predicted_label = row['y_pred']

    # Assign values based on TP, FP, TN, FN
    if true_label == predicted_label:  # Correct prediction (TP or TN)
        dummy_variable.append(1 if true_label == 1 else 4)  # Assign 1 for TP, 4 for TN
    else:  # Incorrect prediction (FP or FN)
        dummy_variable.append(2 if predicted_label == 1 else 3)  # Assign 2 for FP, 3 for FN

# Convert the list to a numpy array
dummy_variable = np.array(dummy_variable)

# Create a new column in the DataFrame
df['Dummy_Variable'] = dummy_variable
df_merged = pd.merge(df1, df, left_index=True, right_index=True)
# Print the DataFrame
print("DataFrame with Dummy Variable:\n", df_merged)
average_quality_by_dummy = df_merged.groupby('Dummy_Variable')['quality'].mean()

# Print the result
print("Average Quality by Dummy Variable:\n", average_quality_by_dummy)

# SVC
model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('SVC')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# KNN
model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('KNN')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Initialize and fit the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

