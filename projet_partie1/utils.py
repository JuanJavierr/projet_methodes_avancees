import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CANDIDATE_LABEL_COLS = ["acc", "acc_rate", "ln_acc_rate"]
LABEL_COL = "acc"

def load_data():
    df = pd.read_csv("data_final_cleaned.csv")
    df = df.drop(columns=["rue_1", "rue_2", "int_no"])

    # Drop other label columns
    df = df.drop(columns=[c for c in CANDIDATE_LABEL_COLS if c != LABEL_COL])

    return df


def prepare_data(df):
    # One hot encoding
    enc = OneHotEncoder()
    enc.fit(df[["borough"]])
    onehot = enc.transform(df[["borough"]]).toarray()
    df = df.drop(columns=["borough"])
    df = pd.concat([df, pd.DataFrame(onehot, columns=enc.get_feature_names_out())], axis=1)

    # Scaler
    scaler = StandardScaler()
    y = df[LABEL_COL]
    df = df.drop(columns=[LABEL_COL])
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    df[LABEL_COL] = y

    return df

def split_data(df, include_val=False):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    X_train = df_train.drop(columns=[LABEL_COL])
    y_train = df_train[LABEL_COL]

    X_test = df_test.drop(columns=[LABEL_COL])
    y_test = df_test[LABEL_COL]

    if include_val:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_train, y_train, X_test, y_test


def evaluate_predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", metrics.mean_squared_error(y_test, y_pred))
    print("R2:", metrics.r2_score(y_test, y_pred))
