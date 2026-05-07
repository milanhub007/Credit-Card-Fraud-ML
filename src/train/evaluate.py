from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print("Confusion Matrix: \n", confusion_matrix(y_test, preds))
    print("Classification report: \n", classification_report(y_test, preds))
