import os
import pickle
import time

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

github_workspace = os.environ.get("GITHUB_WORKSPACE", ".")
tracking_uri_path = os.path.join(github_workspace, "day5", "演習1", "mlruns")
mlflow.set_tracking_uri(tracking_uri_path)
# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists(experiment_name="Titanic", min_runs_for_baseline=1):
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def get_baseline_accuracy(experiment_name="Titanic", min_runs_for_baseline=1):
    """ベースラインの精度を取得"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    experiment_id = experiment.experiment_id

    # まず、ステータスが 'FINISHED' の実行のみを取得
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",  # ステータスのみでフィルタ
        order_by=["attributes.start_time DESC"],
    )

    if runs_df.empty:
        print(f"No finished runs found in experiment '{experiment_name}'.")
        return None

    # 'metrics.accuracy' 列が存在し、かつNaNでない実行をフィルタリング
    # 'metrics.accuracy' がキーとして存在しない場合のエラーを避けるため、まず列の存在を確認
    if "metrics.accuracy" not in runs_df.columns:
        print(
            f"'metrics.accuracy' not found in runs for experiment '{experiment_name}'."
        )
        return None

    filtered_runs = runs_df.dropna(subset=["metrics.accuracy"])

    if filtered_runs.empty or len(filtered_runs) < min_runs_for_baseline:
        print(
            f"Not enough runs with 'metrics.accuracy' ({len(filtered_runs)}) in experiment '{experiment_name}' "
            f"to determine a reliable baseline (minimum: {min_runs_for_baseline})."
        )
        return None

    baseline_run = filtered_runs.iloc[0]  # 最新のものを選択
    baseline_accuracy = baseline_run["metrics.accuracy"]
    print(
        f"Found baseline run: {baseline_run['run_id']} with accuracy: {baseline_accuracy:.4f}"
    )
    return baseline_accuracy


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    current_accuracy = accuracy_score(y_test, y_pred)

    baseline_accuracy = get_baseline_accuracy(experiment_name="Titanic")

    if baseline_accuracy is None:
        pytest.skip(
            "ベースラインとなるモデルの精度が見つからないため、性能劣化テストをスキップします。"
        )
        return

    print(f"Current model accuracy: {current_accuracy:.4f}")
    print(f"Baseline model accuracy: {baseline_accuracy:.4f}")

    # 性能劣化の許容閾値
    # 例1: ベースラインの精度から5%以上低下しない (相対的な低下)
    relative_threshold = 0.95  # ベースラインの95%の精度を維持
    # 例2: ベースラインの精度から絶対値で0.03以上低下しない (絶対的な低下)
    absolute_diff_threshold = 0.03

    assert current_accuracy >= baseline_accuracy * relative_threshold, (
        f"モデルの精度がベースライン ({baseline_accuracy:.4f}) から相対的に許容範囲を超えて低下しました "
        f"(現在値: {current_accuracy:.4f}, 期待値: >= {baseline_accuracy * relative_threshold:.4f})."
    )

    assert current_accuracy >= baseline_accuracy - absolute_diff_threshold, (
        f"モデルの精度がベースライン ({baseline_accuracy:.4f}) から絶対値で許容範囲を超えて低下しました "
        f"(現在値: {current_accuracy:.4f}, 期待値: >= {baseline_accuracy - absolute_diff_threshold:.4f})."
    )

    print("Performance regression test passed.")


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(predictions1, predictions2), (
        "モデルの予測結果に再現性がありません"
    )
