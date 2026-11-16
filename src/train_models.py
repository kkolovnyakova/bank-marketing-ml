
from pathlib import Path
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from features import add_features, make_pipeline

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
for folder in [REPORTS, FIGURES, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
CV = TimeSeriesSplit(n_splits=N_SPLITS)

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0)
}

def evaluate(name, estimator, X, y, use_engineered=True):
    pipeline = make_pipeline(estimator, use_engineered=use_engineered)
    started = time.perf_counter()
    result = cross_validate(
        pipeline, X, y, cv=CV, scoring=SCORING,
        return_train_score=False, n_jobs=1, error_score="raise"
    )
    elapsed = time.perf_counter() - started
    row = {"model": name}
    for metric in SCORING:
        values = result[f"test_{metric}"]
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values, ddof=1))
    row["fit_time_mean_sec"] = float(np.mean(result["fit_time"]))
    row["total_cv_time_sec"] = float(elapsed)
    return row, result

def main():
    train = pd.read_csv(TRAIN_PATH, sep=";")
    y = train["y"].astype(int)
    X = train.drop(columns=["y", "duration"])

    engineered_a = add_features(X)
    engineered_b = add_features(X)
    feature_repeatability = engineered_a.equals(engineered_b)

    preparation_a = make_pipeline(DummyClassifier(strategy="most_frequent"))[:-1]
    preparation_b = make_pipeline(DummyClassifier(strategy="most_frequent"))[:-1]
    preparation_a.fit(X, y)
    preparation_b.fit(X, y)
    transformed_a = preparation_a.transform(X)
    transformed_b = preparation_b.transform(X)
    pipeline_repeatability = bool(np.array_equal(transformed_a, transformed_b))
    repeatability = bool(feature_repeatability and pipeline_repeatability)
    if not repeatability:
        raise RuntimeError("Проверка повторяемости обработки не пройдена")

    models = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression_L1_C1": LogisticRegression(
            C=1.0, penalty="l1", solver="liblinear",
            max_iter=3000, random_state=RANDOM_STATE
        ),
        "LogisticRegression_L2_C1": LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear",
            max_iter=3000, random_state=RANDOM_STATE
        ),
        "GaussianNB": GaussianNB(),
        "DecisionTree_depth5": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=5,
            max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05,
            max_depth=2, min_samples_leaf=20, random_state=RANDOM_STATE
        )
    }

    rows = []
    fold_rows = []
    for name, estimator in models.items():
        row, fold_result = evaluate(name, estimator, X, y)
        rows.append(row)
        for fold_index in range(len(fold_result["test_roc_auc"])):
            fold_rows.append({
                "model": name,
                "fold": fold_index + 1,
                "roc_auc": float(fold_result["test_roc_auc"][fold_index]),
                "pr_auc": float(fold_result["test_pr_auc"][fold_index]),
                "precision": float(fold_result["test_precision"][fold_index]),
                "recall": float(fold_result["test_recall"][fold_index]),
                "f1": float(fold_result["test_f1"][fold_index]),
                "fit_time_sec": float(fold_result["fit_time"][fold_index])
            })

    experiments = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    experiments.to_csv(REPORTS / "experiments.csv", index=False, encoding="utf-8-sig")

    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    regularization_rows = []
    for penalty in ["l1", "l2"]:
        for c_value in c_values:
            estimator = LogisticRegression(
                C=c_value, penalty=penalty, solver="liblinear",
                max_iter=3000, random_state=RANDOM_STATE
            )
            pipeline = make_pipeline(estimator)
            cv_result = cross_validate(
                pipeline, X, y, cv=CV,
                scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                n_jobs=1, error_score="raise"
            )
            fitted = clone(pipeline).fit(X, y)
            coefficients = fitted.named_steps["model"].coef_.ravel()
            regularization_rows.append({
                "penalty": penalty,
                "C": c_value,
                "roc_auc_mean": float(np.mean(cv_result["test_roc_auc"])),
                "roc_auc_std": float(np.std(cv_result["test_roc_auc"], ddof=1)),
                "pr_auc_mean": float(np.mean(cv_result["test_pr_auc"])),
                "nonzero_coefficients": int(np.sum(np.abs(coefficients) > 1e-8))
            })

    regularization = pd.DataFrame(regularization_rows)
    regularization.to_csv(REPORTS / "regularization.csv", index=False, encoding="utf-8-sig")

    depth_rows = []
    for depth in [1, 2, 3, 4, 5, 7, 10, 15, None]:
        estimator = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=20, random_state=RANDOM_STATE
        )
        pipeline = make_pipeline(estimator)
        cv_result = cross_validate(
            pipeline, X, y, cv=CV, scoring="roc_auc",
            return_train_score=True, n_jobs=1, error_score="raise"
        )
        depth_rows.append({
            "max_depth": "None" if depth is None else depth,
            "train_roc_auc_mean": float(np.mean(cv_result["train_score"])),
            "validation_roc_auc_mean": float(np.mean(cv_result["test_score"])),
            "validation_roc_auc_std": float(np.std(cv_result["test_score"], ddof=1))
        })
    depth_results = pd.DataFrame(depth_rows)
    depth_results.to_csv(REPORTS / "tree_depth.csv", index=False, encoding="utf-8-sig")

    ablation_rows = []
    ablation_models = {
        "LogisticRegression_L2": LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear",
            max_iter=3000, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05,
            max_depth=2, min_samples_leaf=20, random_state=RANDOM_STATE
        )
    }
    for model_name, estimator in ablation_models.items():
        for engineered in [False, True]:
            pipeline = make_pipeline(estimator, use_engineered=engineered)
            cv_result = cross_validate(
                pipeline, X, y, cv=CV,
                scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                n_jobs=1, error_score="raise"
            )
            ablation_rows.append({
                "model": model_name,
                "new_features": engineered,
                "roc_auc_mean": float(np.mean(cv_result["test_roc_auc"])),
                "roc_auc_std": float(np.std(cv_result["test_roc_auc"], ddof=1)),
                "pr_auc_mean": float(np.mean(cv_result["test_pr_auc"]))
            })
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(REPORTS / "feature_ablation.csv", index=False, encoding="utf-8-sig")

    logistic_pipeline = make_pipeline(LogisticRegression(
        penalty="l2", solver="liblinear", max_iter=3000,
        random_state=RANDOM_STATE
    ))
    logistic_search = GridSearchCV(
        logistic_pipeline,
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        scoring="roc_auc", cv=CV, n_jobs=1, refit=True
    )
    logistic_search.fit(X, y)

    nb_pipeline = make_pipeline(GaussianNB())
    nb_search = GridSearchCV(
        nb_pipeline,
        {"model__var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]},
        scoring="roc_auc", cv=CV, n_jobs=1, refit=True
    )
    nb_search.fit(X, y)


    tuning = pd.DataFrame([
        {
            "model": "LogisticRegression_L2",
            "best_params": json.dumps(logistic_search.best_params_, ensure_ascii=False),
            "best_cv_roc_auc": float(logistic_search.best_score_)
        },
        {
            "model": "GaussianNB",
            "best_params": json.dumps(nb_search.best_params_, ensure_ascii=False),
            "best_cv_roc_auc": float(nb_search.best_score_)
        }
    ])
    tuning.to_csv(REPORTS / "hyperparameter_search.csv", index=False, encoding="utf-8-sig")

    tuned_candidates = {
        "LogisticRegression_L2_tuned": logistic_search.best_estimator_,
        "GaussianNB_tuned": nb_search.best_estimator_
    }
    tuned_rows = []
    for tuned_name, tuned_pipeline in tuned_candidates.items():
        tuned_result = cross_validate(
            tuned_pipeline, X, y, cv=CV, scoring=SCORING,
            return_train_score=False, n_jobs=1, error_score="raise"
        )
        tuned_row = {"model": tuned_name}
        for metric in SCORING:
            values = tuned_result[f"test_{metric}"]
            tuned_row[f"{metric}_mean"] = float(np.mean(values))
            tuned_row[f"{metric}_std"] = float(np.std(values, ddof=1))
        tuned_row["fit_time_mean_sec"] = float(np.mean(tuned_result["fit_time"]))
        tuned_row["total_cv_time_sec"] = float(np.sum(tuned_result["fit_time"] + tuned_result["score_time"]))
        tuned_rows.append(tuned_row)
        for fold_index in range(len(tuned_result["test_roc_auc"])):
            fold_rows.append({
                "model": tuned_name,
                "fold": fold_index + 1,
                "roc_auc": float(tuned_result["test_roc_auc"][fold_index]),
                "pr_auc": float(tuned_result["test_pr_auc"][fold_index]),
                "precision": float(tuned_result["test_precision"][fold_index]),
                "recall": float(tuned_result["test_recall"][fold_index]),
                "f1": float(tuned_result["test_f1"][fold_index]),
                "fit_time_sec": float(tuned_result["fit_time"][fold_index])
            })

    pd.DataFrame(fold_rows).to_csv(REPORTS / "cv_fold_scores.csv", index=False, encoding="utf-8-sig")
    experiments = pd.concat([experiments, pd.DataFrame(tuned_rows)], ignore_index=True)
    experiments = experiments.sort_values("roc_auc_mean", ascending=False)
    experiments.to_csv(REPORTS / "experiments.csv", index=False, encoding="utf-8-sig")

    maximum_auc = float(experiments["roc_auc_mean"].max())
    eligible = experiments[experiments["roc_auc_mean"] >= maximum_auc - 0.005]
    selected_row = eligible.sort_values("fit_time_mean_sec").iloc[0]
    best_name = str(selected_row["model"])

    candidate_pipelines = {
        "DummyClassifier": make_pipeline(models["DummyClassifier"]),
        "LogisticRegression_L1_C1": make_pipeline(models["LogisticRegression_L1_C1"]),
        "LogisticRegression_L2_C1": make_pipeline(models["LogisticRegression_L2_C1"]),
        "GaussianNB": make_pipeline(models["GaussianNB"]),
        "DecisionTree_depth5": make_pipeline(models["DecisionTree_depth5"]),
        "RandomForest": make_pipeline(models["RandomForest"]),
        "GradientBoosting": make_pipeline(models["GradientBoosting"]),
        "LogisticRegression_L2_tuned": logistic_search.best_estimator_,
        "GaussianNB_tuned": nb_search.best_estimator_
    }
    best_pipeline = candidate_pipelines[best_name]
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, MODELS / "model_v1.joblib")

    linear_pipeline = make_pipeline(
        LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear",
            max_iter=3000, random_state=RANDOM_STATE
        )
    )
    linear_pipeline.fit(X, y)
    preprocessor = linear_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = linear_pipeline.named_steps["model"].coef_.ravel()
    coefficient_table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    }).sort_values("coefficient", key=lambda values: values.abs(), ascending=False)
    coefficient_table.to_csv(REPORTS / "linear_coefficients.csv", index=False, encoding="utf-8-sig")
    coefficient_top = pd.concat([
        coefficient_table.nlargest(10, "coefficient"),
        coefficient_table.nsmallest(10, "coefficient")
    ]).drop_duplicates().sort_values("coefficient")
    coefficient_top.to_csv(REPORTS / "linear_coefficients_top20.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(10, 7))
    plt.barh(coefficient_top["feature"], coefficient_top["coefficient"])
    plt.xlabel("Коэффициент логистической регрессии")
    plt.title("Наиболее значимые коэффициенты логистической модели")
    plt.tight_layout()
    plt.savefig(FIGURES / "linear_coefficients_top20.png", dpi=180)
    plt.close()

    if best_name == "LogisticRegression_L2_C1":
        best_parameters = {"C": 1.0, "penalty": "l2"}
    elif best_name == "LogisticRegression_L2_tuned":
        best_parameters = {"C": float(logistic_search.best_params_["model__C"]), "penalty": "l2"}
    elif best_name == "GaussianNB_tuned":
        best_parameters = {"var_smoothing": float(nb_search.best_params_["model__var_smoothing"])}
    else:
        best_parameters = best_pipeline.named_steps["model"].get_params()

    metadata = {
        "model_version": "v1",
        "algorithm": best_name,
        "best_parameters": best_parameters,
        "primary_metric": "ROC-AUC",
        "cv_roc_auc": float(selected_row["roc_auc_mean"]),
        "validation": "TimeSeriesSplit, 5 folds, no shuffle",
        "training_rows": int(len(train)),
        "test_rows_reserved_for_pr4": 4941,
        "data_source": "UCI Bank Marketing bank-additional-full.csv",
        "raw_sha256": "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8",
        "excluded_features": ["duration", "y"],
        "engineered_features": [
            "contact_history", "campaign_group", "prior_success",
            "age_group", "contact_month"
        ],
        "repeatability_check": bool(repeatability),
        "repeatability_details": {
            "feature_engineering_deterministic": bool(feature_repeatability),
            "full_preprocessing_identical": bool(pipeline_repeatability),
            "transformed_train_shape": list(transformed_a.shape)
        },
        "software_versions": {
            "python": __import__("sys").version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__
        },
        "random_state": RANDOM_STATE,
        "selection_rule": "Максимальный средний ROC-AUC; при разнице менее 0.005 выбирается более быстрая модель",
        "best_observed_roc_auc": maximum_auc,
        "selected_within_005_band": best_name in eligible["model"].values,
        "selection_reason": (
            f"Максимальный ROC-AUC: {maximum_auc:.4f}. Выбрана модель {best_name} "
            f"с ROC-AUC {float(selected_row['roc_auc_mean']):.4f}; разница находится "
            "в пределах правила 0.005, выбран более быстрый вариант."
        ),
        "test_data_usage": "Тестовая выборка не использовалась в ПР3 и сохранена для ПР4."
    }
    with open(MODELS / "model_v1.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    plt.figure(figsize=(9, 5))
    for penalty in ["l1", "l2"]:
        subset = regularization[regularization["penalty"] == penalty]
        plt.plot(subset["C"], subset["roc_auc_mean"], marker="o", label=f"L{1 if penalty == 'l1' else 2}")
    plt.xscale("log")
    plt.xlabel("Параметр C")
    plt.ylabel("Средний ROC-AUC")
    plt.title("Влияние регуляризации на качество логистической регрессии")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "regularization_auc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for penalty in ["l1", "l2"]:
        subset = regularization[regularization["penalty"] == penalty]
        plt.plot(subset["C"], subset["nonzero_coefficients"], marker="o", label=f"L{1 if penalty == 'l1' else 2}")
    plt.xscale("log")
    plt.xlabel("Параметр C")
    plt.ylabel("Число ненулевых коэффициентов")
    plt.title("Влияние регуляризации на число коэффициентов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "regularization_coefficients.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    x_labels = depth_results["max_depth"].astype(str)
    plt.plot(x_labels, depth_results["train_roc_auc_mean"], marker="o", label="Обучающая выборка")
    plt.plot(x_labels, depth_results["validation_roc_auc_mean"], marker="o", label="Проверочные блоки")
    plt.xlabel("Максимальная глубина дерева")
    plt.ylabel("ROC-AUC")
    plt.title("Зависимость качества дерева решений от глубины")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "tree_depth.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    ordered = experiments.sort_values("roc_auc_mean")
    plt.barh(ordered["model"], ordered["roc_auc_mean"], xerr=ordered["roc_auc_std"])
    plt.xlabel("Средний ROC-AUC")
    plt.title("Сравнение моделей по временной кросс-валидации")
    plt.tight_layout()
    plt.savefig(FIGURES / "model_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    ablation["label"] = ablation["model"] + ablation["new_features"].map({True: " + новые признаки", False: " без новых признаков"})
    plt.barh(ablation["label"], ablation["roc_auc_mean"], xerr=ablation["roc_auc_std"])
    plt.xlabel("Средний ROC-AUC")
    plt.title("Оценка влияния новых признаков")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_ablation.png", dpi=180)
    plt.close()

    summary = [
        f"Исходный датасет: 41188 строк; SHA-256: {metadata['raw_sha256']}",
        f"Исходная часть: 24712 строк; дубликатов удалено: 7; после очистки: 24705",
        f"Обучающая часть: {len(train)} строк; тест для ПР4: 4941 строка (не использовался)",
        "Валидация: TimeSeriesSplit, 5 блоков, без перемешивания",
        "Исключены признаки: duration и y",
        "Новые признаки: contact_history, campaign_group, prior_success, age_group, contact_month",
        f"Итоговая модель: {best_name}; ROC-AUC: {float(selected_row['roc_auc_mean']):.4f}",
        f"Правило выбора: {metadata['selection_rule']}",
        f"Проверка повторяемости: {repeatability}"
    ]
    (REPORTS / "run_summary.txt").write_text("\\n".join(summary), encoding="utf-8")

    print(experiments.to_string(index=False))
    print("Повторяемость расчета новых признаков:", feature_repeatability)
    print("Повторяемость полного преобразования:", pipeline_repeatability)
    print("Итоговая модель:", best_name, float(selected_row["roc_auc_mean"]))
    print("Итоговая модель сохранена:", MODELS / "model_v1.joblib")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
