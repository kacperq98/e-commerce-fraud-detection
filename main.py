#|--------------------------------------------------------------|
#|                          Requirements                        |
#|--------------------------------------------------------------|

import logging
from src.common import check_if_path_exists,make_logs_prettier
from src.data_loader import DataLoader
from src.exploratory_data_analysis import ExplaratoryDataAnalysis
from src.feature_engineering import FeatureEngineering
from src.model_training import DataPreparation, XGBoostModelTraining
from src.model_training import XGBoostModelEvaluation, XGBoostModelOptimization, LogisticRegressionBenchmark

#|--------------------------------------------------------------|
#|                          Macors                              |
#|--------------------------------------------------------------|

LOG_DIR = "logs/"
CREATE_HTML_LOGS = True

#|--------------------------------------------------------------|
#|                          Logger                              |
#|--------------------------------------------------------------|

logger = logging.getLogger(__name__)
check_if_path_exists(folder_path=LOG_DIR)
file_handler = logging.FileHandler(f'{LOG_DIR}/pipeline.log', mode = 'w')
file_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(),
                file_handler])


#|--------------------------------------------------------------|
#|                          Main part                           |
#|--------------------------------------------------------------|

class Pipeline:
    def __init__(self):
        pass

    def run(self):
        make_logs_prettier("Pipeline Initialized")
        
        ## STEP 1: DATA LOADING
        make_logs_prettier("STEP 1: DATA LOADING")
        dataset = DataLoader().run_data_loader()
        
        ## STEP 2: EXPLORATORY DATA ANALYSIS (BEFORE FEATURE ENGINEERING)
        make_logs_prettier("STEP 2: EXPLORATORY DATA ANALYSIS (Before Feature Engineering)")
        eda = ExplaratoryDataAnalysis(dataset)
        eda.view_into_data() 
        eda.get_plots(folder_name="BeforeFeatureEngineering")
        eda.create_report(generate_html_report = CREATE_HTML_LOGS, title="Dataset report before Feature Engineering") 
        
        ## STEP 3: FEATURE ENGINEERING
        make_logs_prettier("STEP 3: FEATURE ENGINEERING")
        fe = FeatureEngineering(dataset)
        dataset = fe.run_feature_engineering()

        ## STEP 4: EXPLORATORY DATA ANALYSIS (AFTER FEATURE ENGINEERING)
        make_logs_prettier("STEP 4: EXPLORATORY DATA ANALYSIS (After Feature Engineering)")
        eda = ExplaratoryDataAnalysis(dataset)
        eda.create_report(generate_html_report = CREATE_HTML_LOGS, title="Dataset report after Feature Engineering")
        
        ## STEP 5: DATA PREPARATION
        make_logs_prettier("STEP 5: DATA PREPARATION")
        dp = DataPreparation(dataset)
        prepared_data = dp.run_data_preparing()
        
        ## STEP 6: BENCHMARK MODEL - LOGISTIC REGRESSION
        make_logs_prettier("STEP 6: BENCHMARK MODEL (Logistic Regression)")
        benchmark_model = LogisticRegressionBenchmark(prepared_data)
        benchmark_model.train()
        benchmark_model_metrics = benchmark_model.evaluate()
        benchmark_model.save_model()
        
        ## STEP 7: MODEL TRAINING (XGBoost)
        make_logs_prettier("STEP 7: MODEL TRAINING (XGBoost)")
        trainer = XGBoostModelTraining(prepared_data)
        model = trainer.train_xgboost()
        trainer.save_model("xgboost_baseline.pkl")
        
        ## STEP 8: MODE EVALUATION (XGBoost)
        make_logs_prettier("STEP 8: MODEL EVALUATION (XGBoost)")
        evaluator = XGBoostModelEvaluation(model, prepared_data)
        metrics = evaluator.run_full_evaluation()
        
        ## STEP 9: CROSS-VALIDATION EVALUATION
        make_logs_prettier("STEP 9: CROSS-VALIDATION EVALUATION")
        cv_results = evaluator.evaluate_with_cross_validation(cv=5)
        
        ## STEP 10: MODEL INTERPRETATION (SHAP)
        make_logs_prettier("STEP 10: MODEL INTERPRETATION")
        evaluator.interpret_model_shap()
        
        # ## STEP 11: HYPERPARAMETER OPTIMIZATION
        make_logs_prettier("STEP 11: HYPERPARAMETER OPTIMIZATION")
        optimizer = XGBoostModelOptimization(prepared_data)
        best_params = optimizer.optimize_hyperparameters()
        optimizer.save_optimization_results()
        
        # ## STEP 12: TRAINING OPTIMIZED MODEL
        make_logs_prettier("STEP 12: TRAINING OPTIMIZED MODEL")
        optimized_trainer = optimizer.train_optimized_model(prepared_data)
        optimized_model = optimized_trainer.get_model()
        optimized_trainer.save_model("xgboost_optimized.pkl")
        
        # ## STEP 13: EVALUATING OPTIMIZED MODEL
        make_logs_prettier("STEP 13: EVALUATING OPTIMIZED MODEL")
        optimized_evaluator = XGBoostModelEvaluation(optimized_model, prepared_data)
        optimized_metrics = optimized_evaluator._evaluate_model()
        optimized_evaluator.save_metrics("xgboost_optimized_metrics.json")
        
        ## STEP 14: MODEL INTERPRETATION (SHAP)
        make_logs_prettier("STEP 14: MODEL INTERPRETATION")
        evaluator.interpret_model_shap(path="optimized_model")
        
        
        # ## STEP 14: SUMMARY
        make_logs_prettier("WHOLE PIPELINE COMPLETED SUCCESSFULLY")
        make_logs_prettier("Benchmark Model (Logistic Regression):")
        make_logs_prettier(f"  - ROC-AUC:  {benchmark_model_metrics['roc_auc']:.4f}")
        make_logs_prettier("XGBoost Baseline Model:")
        make_logs_prettier(f"  - ROC-AUC:  {metrics['roc_auc']:.4f}")
        make_logs_prettier(f"  - CV Mean ROC-AUC: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        make_logs_prettier("XGBoost Optimized Model:")
        make_logs_prettier(f"  - ROC-AUC:  {optimized_metrics['roc_auc']:.4f}")
        make_logs_prettier(f"Best Hyperparameters: {best_params}")

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
