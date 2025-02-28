from sklearn.metrics import classification_report

def evaluate_model(model, X_test_vectorized, y_test, class_names):
    y_pred = model.predict(X_test_vectorized)
    print(classification_report(y_test, y_pred, target_names=class_names))
