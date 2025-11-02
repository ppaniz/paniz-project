# heavy_data_science_pipeline.py
# اجرا: python heavy_data_science_pipeline.py
# نیازمند: numpy, pandas, scikit-learn, lightgbm, catboost, xgboost, optuna, shap, joblib

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import optuna
import shap
import joblib
import os
import random

random.seed(42)
np.random.seed(42)


# -----------------------------
# 1) داده‌ی مصنوعی (ترکیبی عددی + دسته‌ای + missing + noise)
# -----------------------------
def make_mixed_dataset(n_samples=20000, n_num=25, n_cat=6, random_state=42):
    X_num, y = make_classification(n_samples=n_samples, n_features=n_num, n_informative=12,
                                   n_redundant=5, n_clusters_per_class=2, flip_y=0.02,
                                   class_sep=1.0, random_state=random_state)
    X_num = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(n_num)])
    # create categorical features by binning and adding categories
    X_cat = pd.DataFrame()
    for i in range(n_cat):
        # bin a numeric col to create categories, then shuffle labels
        col = X_num.iloc[:, i] if i < n_num else X_num.iloc[:, 0]
        cats = pd.qcut(col, q=5, labels=False, duplicates='drop').astype(str)
        # add some new categories and noise
        cats = cats.apply(lambda v: f"cat_{i}_" + str(int(v)))
        # randomly introduce unseen categories
        mask = np.random.rand(len(cats)) < 0.02
        cats.loc[mask] = f"cat_{i}_UNK"
        X_cat[f"cat_{i}"] = cats
    X = pd.concat([X_num, X_cat], axis=1)
    # introduce missingness
    for col in X.columns:
        mask = np.random.rand(len(X)) < 0.05
        X.loc[mask, col] = np.nan
    # add a high-cardinality feature
    X['high_card'] = ["hc_" + str(np.random.randint(0, 500)) for _ in range(len(X))]
    # add some date-like feature (ordinal)
    X['day_of_month'] = np.random.randint(1, 29, size=len(X))
    return X, y


X, y = make_mixed_dataset()
print("Dataset shape:", X.shape)


# -----------------------------
# 2) Custom transformer examples (target encoding approximation + interactions)
# -----------------------------
class SimpleTargetEncoder(BaseEstimator, TransformerMixin):
    """very simple blend target encoder (fits on training only)"""

    def __init__(self, cols=None, smoothing=20):
        self.cols = cols
        self.smoothing = smoothing
        self.target_stats = {}

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y)
        for col in self.cols:
            stats = X.groupby(col)[0].agg(['count']).rename(columns={'count': 'n'})
            # compute mean target per category from original training X (we'll handle externally)
        # We'll compute in transform because we need access to full arrays
        return self

    def transform(self, X, y=None):
        # Here we implement a simple frequency-based encoding using global mean and category mean estimated from provided y if available.
        X = pd.DataFrame(X).copy()
        if y is None:
            # Inference: just use category frequency mapping produced earlier (not perfect but ok for demo)
            for col in self.cols:
                X[col + "_te"] = X[col].map(self._map.get(col, {})).fillna(self.global_mean)
            return X
        # Fit+transform mode
        y = pd.Series(y)
        self.global_mean = y.mean()
        self._map = {}
        for col in self.cols:
            df = pd.concat([X[col], y], axis=1)
            stats = df.groupby(col)[y.name].agg(['mean', 'count']).rename(columns={'mean': 'mean', 'count': 'n'})
            # smoothing
            stats['smooth'] = (stats['mean'] * stats['n'] + self.global_mean * self.smoothing) / (
                    stats['n'] + self.smoothing)
            self._map[col] = stats['smooth'].to_dict()
        for col in self.cols:
            X[col + "_te"] = X[col].map(self._map.get(col, {})).fillna(self.global_mean)
        return X


# -----------------------------
# 3) Column lists & preprocessing pipelines
# -----------------------------
numeric_cols = [c for c in X.columns if c.startswith("num_")]
categorical_cols = [c for c in X.columns if c.startswith("cat_")]
hc_col = ['high_card']
ordinal_cols = ['day_of_month']

# numeric pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# categorical pipeline (one-hot for low-cardinality)
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


# high-card pipeline: use simple ordinal + freq + leave as is
class HighCardFeats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.Series(X.iloc[:, 0])
        self.top_k = X.value_counts().index[:100]  # keep top100 explicit
        return self

    def transform(self, X):
        X = pd.Series(X.iloc[:, 0]).copy()
        X2 = pd.DataFrame()
        # frequency
        vc = X.value_counts(normalize=True)
        X2['high_card_freq'] = X.map(vc).fillna(0)
        # top_k indicator
        X2['high_card_top'] = X.isin(self.top_k).astype(int)
        return X2


hc_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='hc_MISSING')),
    ('feats', HighCardFeats())
])

# ordinal pipeline
ord_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('pass', 'passthrough')
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numeric_cols),
    ('cat', cat_pipeline, categorical_cols),
    ('hc', hc_pipeline, hc_col),
    ('ord', ord_pipeline, ordinal_cols)
], remainder='drop')

# -----------------------------
# 4) Train-test split (holdout)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, stratify=y, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)


# -----------------------------
# 5) Objective for Optuna to tune LightGBM (as base example)
# -----------------------------
def objective_lgb(trial, X, y):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    }
    # build pipeline inside objective
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', lgb.LGBMClassifier(**param, n_estimators=1000, random_state=42, n_jobs=1))
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in cv.split(X, y):
        Xtr, Xv = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yv = y[tr_idx], y[val_idx]
        pipe.fit(Xtr, ytr)
        preds = pipe.predict_proba(Xv)[:, 1]
        scores.append(roc_auc_score(yv, preds))
    return np.mean(scores)


# -----------------------------
# 6) Run Optuna to get best LGB params (nested CV-like)
# -----------------------------
study = optuna.create_study(direction='maximize', study_name='lgb_auc_study')
# use a smaller number of trials for speed; increase for real heavy tuning
study.optimize(lambda t: objective_lgb(t, X_train, y_train), n_trials=30, n_jobs=1)
print("Best optuna LGB AUC:", study.best_value)
best_params = study.best_trial.params
print("Best params:", best_params)

# -----------------------------
# 7) Build base learners with tuned-ish params
# -----------------------------
lgb_clf = Pipeline([
    ('pre', preprocessor),
    ('lgbm', lgb.LGBMClassifier(n_estimators=1500, random_state=42, **best_params, n_jobs=-1))
])

cat_clf = Pipeline([
    ('pre', preprocessor),
    ('cat', CatBoostClassifier(iterations=800, learning_rate=0.05, eval_metric='AUC', verbose=0, random_state=42))
])

xgb_clf = Pipeline([
    ('pre', preprocessor),
    ('xgb', xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, use_label_encoder=False, eval_metric='auc',
                              random_state=42, n_jobs=-1))
])

# -----------------------------
# 8) Stacking ensemble
# -----------------------------
estimators = [
    ('lgb', lgb_clf),
    ('cat', cat_clf),
    ('xgb', xgb_clf)
]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1,
                           passthrough=False)

# cross-validate stacking quickly to see baseline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for tr_idx, val_idx in cv.split(X_train, y_train):
    Xtr, Xv = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    ytr, yv = y_train[tr_idx], y_train[val_idx]
    stack.fit(Xtr, ytr)
    preds = stack.predict_proba(Xv)[:, 1]
    sc = roc_auc_score(yv, preds)
    scores.append(sc)
    print("Fold AUC:", sc)
print("Stacking CV AUC mean:", np.mean(scores))

# -----------------------------
# 9) Fit on full train and evaluate on holdout test
# -----------------------------
print("Fitting stacking on full train set (this may take a while)...")
stack.fit(X_train, y_train)
preds_test = stack.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, preds_test)
print("Holdout Test AUC:", test_auc)

# -----------------------------
# 10) SHAP explainability for one base model (lgb) after transforming
# -----------------------------
# extract preprocessor and lgb model separately for SHAP
# We'll fit a standalone lgb pipeline to get feature names
print("Fitting a standalone LGB for SHAP (fast fit)...")
lgb_pipe_for_shap = Pipeline(
    [('pre', preprocessor), ('lgbm', lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42))])
lgb_pipe_for_shap.fit(X_train, y_train)
# get transformed features
X_train_trans = lgb_pipe_for_shap.named_steps['pre'].transform(X_train)
feature_names = []
# build names from transformer (approximate)
# numeric names
feature_names += numeric_cols
# onehot applied names
oh = lgb_pipe_for_shap.named_steps['pre'].named_transformers_['cat'].named_steps['onehot']
cat_ohe_names = oh.get_feature_names_out(categorical_cols).tolist()
feature_names += cat_ohe_names
# high-card features
feature_names += ['high_card_freq', 'high_card_top']
# ordinal
feature_names += ordinal_cols

print("Num transformed features:", X_train_trans.shape[1], "Feature names len:", len(feature_names))
# SHAP expects original model reference
explainer = shap.TreeExplainer(lgb_pipe_for_shap.named_steps['lgbm'])
print("Calculating SHAP values (sampled)...")
sample_idx = np.random.choice(len(X_train_trans), size=min(2000, len(X_train_trans)), replace=False)
X_sample = X_train_trans[sample_idx]
shap_values = explainer.shap_values(X_sample)
# Summarize top features
shap_abs_mean = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(shap_abs_mean)[-12:][::-1]
print("Top SHAP features:")
for i in top_idx:
    name = feature_names[i] if i < len(feature_names) else f"f{i}"
    print(f"{name} -> mean(|shap|)={shap_abs_mean[i]:.6f}")

# -----------------------------
# 11) Persist models & artifacts
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(stack, "models/stacking_ensemble.joblib")
joblib.dump(lgb_pipe_for_shap, "models/lgb_pipeline_for_shap.joblib")
print("Saved models to models/*.joblib")


# -----------------------------
# 12) Quick function to predict on new data
# -----------------------------
def predict_proba_dataframe(df):
    model = joblib.load("models/stacking_ensemble.joblib")
    return model.predict_proba(df)[:, 1]


# -----------------------------
# 13) Final short report
# -----------------------------
print("--- final report ---")
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Test AUC:", test_auc)
print("Models saved at ./models")

# optional: if you want to inspect SHAP summary plot, uncomment below (requires matplotlib)
# import matplotlib.pyplot as plt
# shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
