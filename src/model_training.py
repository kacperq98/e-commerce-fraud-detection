#|--------------------------------------------------------------|
#|                          Requirements                        |
#|--------------------------------------------------------------|

import logging
import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import optuna.visualization as ov

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

import src.common as common

#|--------------------------------------------------------------|
#|                          Macors                              |
#|--------------------------------------------------------------|

MODELS_PATH = "models/"
PLOTS_PATH = "plots/ModelEvaluation/"
METRICS_PATH = "metrics/"
OPTUNA_PLOTS_PATH = "plots/Optuna/"
PCA_COMPONENTS = 10

#|--------------------------------------------------------------|
#|                          Main part                           |
#|--------------------------------------------------------------|

logger = logging.getLogger(__name__)

## SECTION 1: DATAPREPARATION
class DataPreparation:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = self.dataset.drop(columns=['is_fraud'])
        self.y = self.dataset['is_fraud']
        self.scaler = StandardScaler()
        self.feature_names = list(self.X.columns)

    def _train_test_split(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y)

        train_test_dict = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test}

        return train_test_dict

    def _normalization(self, train_test_dict):
        X_train_scaled = self.scaler.fit_transform(train_test_dict["X_train"])
        X_test_scaled = self.scaler.transform(train_test_dict["X_test"])

        train_test_scaled_dict = {
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled}

        return train_test_scaled_dict

    def _perform_pca(self, n_components = 5):
        pca = PCA(n_components = n_components)
        X_pca = pca.fit_transform(self.X)

        return X_pca

    def run_data_preparing(self, test_size=0.2, random_state=42):
        train_test_dict = self._train_test_split(test_size=test_size, random_state=random_state)
        train_test_scaled_dict = self._normalization(train_test_dict)
        X_pca = self._perform_pca(n_components = PCA_COMPONENTS)

        fraud_count = (train_test_dict["y_train"] == 1).sum()
        non_fraud_count = (train_test_dict["y_train"] == 0).sum()
        scale_pos_weight = non_fraud_count / fraud_count if fraud_count > 0 else 1

        logger.info(f"\nClass distribution - Non-Fraud: {non_fraud_count}, Fraud: {fraud_count}")
        logger.info(f"\nScale pos weight (for imbalanced data): {scale_pos_weight:.2f}")

        prepared_data = {
            "X_train": train_test_dict["X_train"],
            "X_test": train_test_dict["X_test"],
            "y_train": train_test_dict["y_train"],
            "y_test": train_test_dict["y_test"],
            "X_pca": X_pca,
            "scale_pos_weight": scale_pos_weight,
            "feature_names": self.feature_names,
            "scaler": self.scaler}

        return prepared_data


## SECTION 2A: XGBOOST MODEL TRAINING
class XGBoostModelTraining:
    def __init__(self, prepared_data):
        self.X_train = prepared_data["X_train"]
        self.X_test = prepared_data["X_test"]
        self.y_train = prepared_data["y_train"]
        self.y_test = prepared_data["y_test"]
        self.scale_pos_weight = prepared_data["scale_pos_weight"]
        self.feature_names = prepared_data["feature_names"]
        self.pca_n_components = PCA_COMPONENTS
        self.model = None

    def train_xgboost(self, params=None):
        if params is None:
            params = {'n_estimators': 8,
                    'max_depth': 4,
                    'learning_rate': 0.3,
                    'scale_pos_weight': self.scale_pos_weight,
                    'random_state': 42,
                    'lambda': 0.8,
                    'alpha': 0.2,
                    'eval_metric': 'logloss',
                    'base_score': 0.5}
        
        logger.info(f"Training XGBoost with parameters: {params}")

        self.pipeline = make_pipeline(StandardScaler(), PCA(n_components = self.pca_n_components), XGBClassifier(**params))
        self.pipeline.fit(self.X_train, self.y_train)

        train_score = self.pipeline.score(self.X_train, self.y_train)
        logger.info(f"XGBoost model training completed.. Train accuracy: {train_score:.4f}")

        self.model = self.pipeline
        return self.pipeline

    def save_model(self, filename="xgboost_model.pkl"):
        common.check_if_path_exists(MODELS_PATH)
        filepath = os.path.join(MODELS_PATH, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filename="xgboost_model.pkl"):
        filepath = os.path.join(MODELS_PATH, filename)
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return self.model

    def get_model(self):
        return self.model


## SECTION 2B: LOGISTIC REGRESSION MODEL TRAINING (BENCHMARK MODEL)
class LogisticRegressionBenchmark:
    def __init__(self, prepared_data):
        self.X_train = prepared_data["X_train"]
        self.X_test = prepared_data["X_test"]
        self.y_train = prepared_data["y_train"]
        self.y_test = prepared_data["y_test"]
        self.feature_names = prepared_data["feature_names"]
        self.model = None
        self.pca_n_components = PCA_COMPONENTS

    def train(self, max_iter=1000, class_weight='balanced'):
        logger.info("Training Logistic Regression benchmark model...")
        
        self.model = make_pipeline(
            PCA(n_components=self.pca_n_components, random_state=42),
            LogisticRegression(max_iter=max_iter, class_weight=class_weight,
                            random_state=42, penalty='l2'))

        self.model.fit(self.X_train, self.y_train)

        pca_fitted = self.model.named_steps['pca'] 
        n_components_used = pca_fitted.n_components_

        train_accuracy = self.model.score(self.X_train, self.y_train)

        logger.info(f"PCA reduced dimensions from {self.X_train.shape[1]} to {n_components_used}.")
        logger.info(f"Logistic Regression training completed. Train accuracy: {train_accuracy:.4f}")

        return self.model

    def evaluate(self):
        logger.info("Evaluating Logistic Regression benchmark mdel...")

        y_pred = self.model.predict(self.X_test)
        y_pred_probability = self.model.predict_proba(self.X_test)[:, 1]

        metrics = {'accuracy': accuracy_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_probability)}

        logger.info("LOGISTIC REGRESSION BENCHMARK RESULTS")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

        report = classification_report(self.y_test, y_pred)

        logger.info("\nClassification Report:\n" + report)
        metrics['classification_report'] = classification_report(self.y_test, y_pred, output_dict=True)

        return metrics

    def get_model(self):
        return self.model

    def save_model(self, filename="logistic_regression_benchmark.pkl"):
        common.check_if_path_exists(MODELS_PATH)
        filepath = os.path.join(MODELS_PATH, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_model(), f)
        logger.info(f"Benchmark model saved to {filepath}")
        return filepath
    
    
## SECTION 3: MODEL EVALUATION
class XGBoostModelEvaluation:
    def __init__(self, model, prepared_data):
        self.model = model
        self.model_named_steps = model.named_steps['xgbclassifier']
        self.X_train = prepared_data["X_train"]
        self.X_test = prepared_data["X_test"]
        self.y_train = prepared_data["y_train"]
        self.y_test = prepared_data["y_test"]
        self.feature_names = prepared_data["feature_names"]
        self.metrics = {}

    def _evaluate_model(self):
        logger.info("Evaluating model...")

        y_pred = self.model.predict(self.X_test)
        y_pred_probability = self.model.predict_proba(self.X_test)[:, 1]
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)

        self.metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred), #(TP + TN) / (TP + TN + FP + FN}
            'roc_auc': roc_auc_score(self.y_test, y_pred_probability),
            'precision': precision, #TP / (TP + FP),
            'recall': recall} #TP / (TP + FN)

        logger.info("MODEL EVALUATION RESULTS")
        logger.info(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        logger.info(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")

        report = classification_report(self.y_test, y_pred)
        logger.info("\nClassification Report:\n" + report)
        self.metrics['classification_report'] = classification_report(self.y_test, y_pred, output_dict=True)

        self.y_pred = y_pred
        self.y_pred_probability = y_pred_probability

        return self.metrics

    def _plot_confusion_matrix(self):
        common.check_if_path_exists(PLOTS_PATH)

        cm = confusion_matrix(self.y_test, self.y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Fraud', 'Fraud'],
                    yticklabels=['Non-Fraud', 'Fraud'])
        plt.title('Confusion Matrix - XGBoost')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        filepath = os.path.join(PLOTS_PATH, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {filepath}")


    def _plot_feature_importance(self):

        importance = self.model_named_steps.feature_importances_
        indices = np.argsort(importance)[::-1][:20]

        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance - XGBoost (Top 20)')
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        filepath = os.path.join(PLOTS_PATH, 'feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to {filepath}")

    def _plot_permutation_importance(self, n_repeats=30):
        common.check_if_path_exists(PLOTS_PATH)
        logger.info("Calculating permutation importance...")

        r = permutation_importance(
            self.model, 
            self.X_test, 
            self.y_test, 
            n_repeats=n_repeats, 
            random_state=42,
            n_jobs=-1)

        sorted_descending_idx = r.importances_mean.argsort()[::-1]
        sorted_descending_idx = sorted_descending_idx[:20]

        plt.figure(figsize=(12, 8))
        plt.barh(
            range(len(sorted_descending_idx)), 
            r.importances_mean[sorted_descending_idx], 
            xerr=r.importances_std[sorted_descending_idx],
            color='maroon',
            align='center')

        plt.yticks(range(len(sorted_descending_idx)), [self.feature_names[i] for i in sorted_descending_idx])
        plt.xlabel('Permutation Importance (decrease in score)')
        plt.ylabel('Features')
        plt.title('Permutation Feature Importance (Top 20)')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        filepath = os.path.join(PLOTS_PATH, 'permutation_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Permutation importance plot saved to {filepath}")

        return r

    def evaluate_with_cross_validation(self, cv=5):
        logger.info(f"Running {cv}-fold cross-validation...")

        X_full = np.vstack([self.X_train, self.X_test])
        y_full = np.concatenate([self.y_train, self.y_test])

        cv_scores = cross_val_score(self.model, X_full, y_full, cv=cv, scoring='roc_auc')

        logger.info("CROSS-VALIDATION RESULTS")
        logger.info(f"Cross-Validation Scores: {cv_scores}")
        logger.info(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
        logger.info(f"Std ROC-AUC: {cv_scores.std():.4f}")
        
        return {'cv_scores': cv_scores.tolist(),
                'mean_score': float(cv_scores.mean()),
                'std_score': float(cv_scores.std())}

    def interpret_model_shap(self, max_samples=1000):
        logger.info("Generating SHAP interpretations...")
        common.check_if_path_exists(PLOTS_PATH)

        try:
            if len(self.X_test) > max_samples:
                sample_indices = np.random.choice(len(self.X_test), max_samples, replace=False)
                X_sample_df = self.X_test.iloc[sample_indices]
            else:
               X_sample_df = self.X_test

            X_sample = X_sample_df.values
            self.model.named_steps['xgbclassifier'] = self.model.named_steps['xgbclassifier'].base_score = 0.5

            explainer = shap.TreeExplainer(self.model.named_steps['xgbclassifier'])
            shap_values = explainer.shap_values(X_sample)

            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            plt.title('SHAP Summary Plot - XGBoost')
            plt.tight_layout()
            filepath = os.path.join(PLOTS_PATH, 'shap_summary.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP summary plot saved to {filepath}")

            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names,
                              plot_type="bar", show=False)
            plt.title('SHAP Feature Importance - XGBoost')
            plt.tight_layout()
            filepath = os.path.join(PLOTS_PATH, 'shap_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP importance plot saved to {filepath}")


            explainer_new = shap.Explainer(self.model, self.X_train[:100].values)
            shap_values_new = explainer_new(X_sample[:100])

            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(shap_values_new[0], max_display=15, show=False)
            plt.title('SHAP Waterfall Plot - Single Prediction Breakdown')
            plt.tight_layout()
            filepath = os.path.join(PLOTS_PATH, 'shap_waterfall.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP waterfall plot saved to {filepath}")

            mean_shap = np.abs(shap_values).mean(axis=0)
            top_feature_idx = np.argmax(mean_shap)
            top_feature_name = self.feature_names[top_feature_idx]

            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                top_feature_name, 
                shap_values, 
                X_sample, 
                feature_names=self.feature_names,
                show=False)

            plt.title(f'SHAP Dependence Plot - {top_feature_name}')
            plt.tight_layout()
            filepath = os.path.join(PLOTS_PATH, f'shap_dependence_{top_feature_name}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP dependence plot saved to {filepath}")

            plt.figure(figsize=(20, 3))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[0], 
                X_sample[0], 
                feature_names=self.feature_names,
                matplotlib=True,
                show=False)

            plt.title('SHAP Force Plot - Single Prediction')
            plt.tight_layout()
            filepath = os.path.join(PLOTS_PATH, 'shap_force_plot.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP force plot saved to {filepath}")

            return shap_values

        except Exception as e:
            logger.warning(f"SHAP interpretation failed (XGBoost/SHAP version compatibility issue): {e}")
            logger.warning("Skipping SHAP plots. Feature importance from XGBoost is still available.")
            return None

    def save_metrics(self, filename="xgboost_metrics.json"):
        common.check_if_path_exists(METRICS_PATH)
        filepath = os.path.join(METRICS_PATH, filename)

        metrics_to_save = {
            'accuracy': float(self.metrics['accuracy']),
            'roc_auc': float(self.metrics['roc_auc'])}

        with open(filepath, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)

        logger.info(f"Metrics saved to {filepath}")
        return filepath
    
    def run_full_evaluation(self):
        self._evaluate_model()
        self._plot_confusion_matrix()
        self._plot_feature_importance()

        try:
            self._plot_permutation_importance()
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")

        self.save_metrics()
        return self.metrics
        
## SECTION 4: MODEL OPTIMIZATION
class XGBoostModelOptimization:
    def __init__(self, prepared_data):
        self.X_train = prepared_data["X_train"]
        self.y_train = prepared_data["y_train"]
        self.scale_pos_weight = prepared_data["scale_pos_weight"]
        self.feature_names = prepared_data["feature_names"]
        self.early_stopping_rounds = 10
        self.best_params = None
        self.best_model = None
        self.study = None

    def optimize_hyperparameters(self, n_trials=50, cv=10, scoring='f1'):
        logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        logger.info(f"Using {cv}-fold cross-validation, optimizing for: {scoring}")

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 30),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 2, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                "alpha": trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                "lambda": trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'scale_pos_weight': self.scale_pos_weight,
                'random_state': 42,
                'eval_metric': 'logloss',}

            model = XGBClassifier(**params)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring)
            return scores.mean()

        self.study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, n_min_trials=10))
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.plot_optuna_history(self.study)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info("OPTUNA OPTIMIZATION RESULTS")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best {scoring} score (CV): {self.best_score:.4f}")
        logger.info(f"Number of finished trials: {len(self.study.trials)}")

        return self.best_params

    def get_best_params(self):
        return self.best_params

    def get_best_model(self):
        return self.best_model

    def get_study(self):
        return self.study

    def train_optimized_model(self, prepared_data):
        logger.info("Training model with optimized parameters...")

        optimized_params = {
            **self.best_params,
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity' : 0}

        trainer = XGBoostModelTraining(prepared_data)
        trainer.train_xgboost(params=optimized_params)

        return trainer
    
    def plot_optuna_history(self, study=None):
        hist_fig = ov.plot_optimization_history(study)
        hist_fig.update_layout(title='Optimization History')
        common.check_if_path_exists(OPTUNA_PLOTS_PATH)
        hist_fig.write_html(f"{OPTUNA_PLOTS_PATH}/optimization_history.html", auto_open=False)

        importances_fig = ov.plot_param_importances(study)
        importances_fig.update_layout(title='Params Importance')
        common.check_if_path_exists(OPTUNA_PLOTS_PATH)
        importances_fig.write_html(f"{OPTUNA_PLOTS_PATH}/Pparams_importance.html", auto_open=False)
        
        key_params = ['max_depth', 'learning_rate']        
        if all(p in study.best_params for p in key_params):
            contour_fig = ov.plot_contour(study, params=key_params)
            contour_fig.update_layout(title=f'{key_params[0]} vs {key_params[1]}')
            common.check_if_path_exists(OPTUNA_PLOTS_PATH)
            importances_fig.write_html(f"{OPTUNA_PLOTS_PATH}/{key_params[0]}_vs_{key_params[1]}.html", auto_open=False)


    def save_optimization_results(self, filename="optimization_results.json"):
        common.check_if_path_exists(METRICS_PATH)
        filepath = os.path.join(METRICS_PATH, filename)

        results = {'best_params': self.best_params,
                    'best_score': float(self.best_score),
                    'n_trials': len(self.study.trials)}

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Optimization results saved to {filepath}")
        return filepath
