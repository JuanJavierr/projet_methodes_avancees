import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(label_col="acc"):
    candidate_label_cols = ["acc", "acc_rate", "ln_acc_rate"]

    if label_col not in candidate_label_cols:
        raise ValueError(f"label_col must be one of {candidate_label_cols}")

    df = pd.read_csv("data_final_cleaned.csv")
    df = df.drop(columns=["rue_1", "rue_2", "int_no"])

    # Drop other label columns
    df = df.drop(columns=[c for c in candidate_label_cols if c != label_col])

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
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df

def split_data(df, label_col, include_val=False):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    X_train = df_train.drop(columns=[label_col])
    y_train = df_train[label_col]

    X_test = df_test.drop(columns=[label_col])
    y_test = df_test[label_col]

    if include_val:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_train, y_train, X_test, y_test
