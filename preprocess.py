import pandas as pd

def preprocess_data(df: pd.DataFrame, training=True, feature_columns=None):
    df = df.copy()

    # Basic feature engineering
    df["TransactionAmt_log"] = df["TransactionAmt"].apply(lambda x: 0 if x <= 0 else np.log(x))

    # Encode ProductCD
    if "ProductCD" in df.columns:
        df = pd.get_dummies(df, columns=["ProductCD"], dummy_na=True)

    # Card features
    card_cols = [col for col in df.columns if "card" in col]
    for col in card_cols:
        df[col] = df[col].astype(str)

    df = pd.get_dummies(df, columns=card_cols, dummy_na=True)

    # Handle V features
    v_cols = [col for col in df.columns if col.startswith("V")]
    missing_ratio = df[v_cols].isnull().mean()
    v_keep = missing_ratio[missing_ratio < 0.5].index.tolist()
    df = df[v_keep + [col for col in df.columns if not col.startswith("V")]]

    df.fillna(-999, inplace=True)

    # Align columns for inference
    if not training:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

    return df