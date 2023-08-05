import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Load the data from the CSV file


def load_data(lines):
    lines = pd.read_csv('D:/folders/ML/CSV/naivedata.csv')
    return lines

# Split the data into features and target


def split_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

# Train and test the Naive Bayes classifier


def train_test_naive_bayes(X_train, y_train, X_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb.predict(X_test)


if __name__ == "__main__":
    # Load data from CSV file
    file_path = 'data.csv'
    data = load_data(file_path)

    # Split data into features and target
    X, y = split_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train and test the Naive Bayes classifier
    y_pred = train_test_naive_bayes(X_train, y_train, X_test)

    # Calculate and display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Calculate and display precision
    precision = precision_score(y_test, y_pred)
    print("Precision:", precision)

    # Calculate and display F1 score
    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)
