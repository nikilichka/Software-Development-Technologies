import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Load the Iris dataset and return it as a pandas DataFrame.
    """
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

def preprocess_data(data):
    """
    Preprocess the dataset by adding a new feature and standardizing numerical features.
    """
    # Add a new feature: the ratio of sepal length to petal length
    data['sepal_petal_ratio'] = data['sepal length (cm)'] / data['petal length (cm)']
    
    # Standardize features (excluding 'target' and 'sepal_petal_ratio')
    scaler = StandardScaler()
    features = data.drop(columns=['target', 'sepal_petal_ratio'])
    data[features.columns] = scaler.fit_transform(features)
    
    # Ensure that the target variable is of type integer
    data['target'] = data['target'].astype(int)
    
    return data

if __name__ == "__main__":
    # Load the Iris dataset
    data = load_data()
    
    # Preprocess the dataset (feature engineering & scaling)
    data = preprocess_data(data)
    
    # Save the preprocessed data to a CSV file
    data.to_csv("cleaned_data.csv", index=False)
    
    print("Preprocessed data saved as 'cleaned_data.csv'.")
