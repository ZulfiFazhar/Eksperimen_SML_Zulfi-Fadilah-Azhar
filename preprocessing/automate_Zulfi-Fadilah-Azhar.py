import argparse
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def run_eda(df: pd.DataFrame, target_col: str):

    na_count = df.isna().sum()
    print("\nMissing per column (desc):")
    print(na_count.sort_values(ascending=False))

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\nCategorical columns unique counts and samples:")
    for c in cat_cols:
        uniq = df[c].unique()
        print(f"- {c}: nunique={df[c].nunique()} sample={uniq[:10]}")

    print("\nTarget distribution:")
    print(df[target_col].value_counts())

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    cat_cols = [c for c in df.columns if c not in num_cols + [target_col]]

    print("\nNumeric:", num_cols)
    print("Categorical:", cat_cols)
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )
    return preprocess


def to_dataframe_after_transform(Xt, preprocess, num_cols, cat_cols):

    if sp.issparse(Xt):
        Xt = Xt.toarray()

    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    final_cols = [f"num__{c}" for c in num_cols] + ohe_feature_names

    Xt_df = pd.DataFrame(Xt, columns=final_cols)
    return Xt_df


def main():

    ap = argparse.ArgumentParser(description="Automate preprocessing to produce a train-ready dataset, mirroring the notebook steps.")
    ap.add_argument("--input", required=True, help="Path input CSV mentah, contoh: '../Credit Score Classification Dataset_raw.csv'")
    ap.add_argument("--output", required=True, help="Path output CSV hasil preprocessing, contoh: 'Credit Score Classification Dataset_preprocessing.csv'")
    ap.add_argument("--target", default="Credit Score", help="Nama kolom target. Default: 'Credit Score'")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    target_col = args.target
    num_cols, cat_cols = run_eda(df, target_col)

    preprocess = build_preprocessor(num_cols, cat_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    Xt = preprocess.fit_transform(X)

    print("\nTransformed matrix preview (first 3 rows):")
    if sp.issparse(Xt):
        print(Xt[:3].toarray())
    else:
        print(Xt[:3])

    Xt_df = to_dataframe_after_transform(Xt, preprocess, num_cols, cat_cols)
    Xt_df[target_col] = y.values

    print("\nTransformed DataFrame head:")
    print(Xt_df.head())

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Xt_df.to_csv(args.output, index=False)
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
