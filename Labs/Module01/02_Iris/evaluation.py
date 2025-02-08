from sklearn.metrics import classification_report
from preprocessing import load_data, preprocess_data
from training import train_model

# Evaluate the model with more details
def evaluate_model(data, model):
    X = data.drop("target", axis=1)
    y = data["target"]

    predictions = model.predict(X)
    report = classification_report(y, predictions)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    data = load_data()
    data = preprocess_data(data)
    model = train_model(data)
    evaluate_model(data, model)
