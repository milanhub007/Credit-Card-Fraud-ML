from src.data.load_data import load_data
from src.validate.validate_data import validate_data
from src.data.preprocess import preprocess
from src.train.train import train_model
from src.train.evaluate import evaluate_model


def ml_pipeline():
    data = load_data()

    df = preprocess(data)

    all_passed, failed, cleaned_data = validate_data(df)

    model, X_test, y_test = train_model(cleaned_data, target_col='Class')

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    ml_pipeline()
