"""
Module 5 Week A — Core Skills Drill: Classification & Evaluation Basics

Complete the three functions below. Each function has a docstring
describing its inputs, outputs, and purpose.

Run your work: python drill.py
Test your work: the autograder runs automatically when you open a PR.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(df, target_col="churned", test_size=0.2, random_state=42):
    """
    Task 1: Split a DataFrame into train and test sets with stratification.
    """
    # فصل الميزات (X) عن عمود الهدف (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # التقسيم مع ضمان توازن الفئات (stratify=y) بنسبة 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def compute_classification_metrics(y_true, y_pred):
    """
    Task 2: Compute classification metrics from true and predicted labels.
    """
    # حساب المقاييس الأربعة المطلوبة في قاموس
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred))
    }
    return metrics


def run_cross_validation(X_train, y_train, n_folds=5, random_state=42):
    """
    Task 3: Run stratified k-fold cross-validation with LogisticRegression.
    """
    # إنشاء نموذج Logistic Regression مع موازنة الفئات
    model = LogisticRegression(
        max_iter=1000, 
        random_state=random_state, 
        class_weight="balanced"
    )
    
    # إعداد التقسيم الطبقي (Stratified) بخمس طيات
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # تشغيل التحقق المتقاطع لحساب الدقة (accuracy)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    
    # تجميع النتائج المطلوبة: القائمة، المتوسط، والانحراف المعياري
    cv_results = {
        'scores': scores,
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores))
    }
    return cv_results


if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_csv("data/telecom_churn.csv")
        print(f"Loaded {len(df)} rows")

        # Task 1: Split
        numeric_cols = ["tenure", "monthly_charges", "total_charges",
                        "num_support_calls", "senior_citizen", "has_partner",
                        "has_dependents"]
        df_numeric = df[numeric_cols + ["churned"]]

        result = split_data(df_numeric)
        if result is not None:
            X_train, X_test, y_train, y_test = result
            print(f"Train: {len(X_train)}, Test: {len(X_test)}")

            # Task 2: Metrics
            # نستخدم class_weight="balanced" هنا أيضاً لمطابقة المهمة 3
            model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_classification_metrics(y_test, y_pred)
            if metrics:
                print(f"Metrics: {metrics}")

            # Task 3: Cross-validation
            cv_results = run_cross_validation(X_train, y_train)
            if cv_results:
                print(f"CV: {cv_results['mean']:.3f} +/- {cv_results['std']:.3f}")
                
    except FileNotFoundError:
        print("Error: Make sure 'data/telecom_churn.csv' exists.")