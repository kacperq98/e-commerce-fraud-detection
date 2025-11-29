#|--------------------------------------------------------------|
#|                          Requirements                        |
#|--------------------------------------------------------------|

from ydata_profiling import ProfileReport
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import src.common as common
import numpy as np
import logging
import os 

#|--------------------------------------------------------------|
#|                          Macors                              |
#|--------------------------------------------------------------|

REPORT_PATH = "report/"

#|--------------------------------------------------------------|
#|                          Main part                           |
#|--------------------------------------------------------------|

logger = logging.getLogger(__name__)
class ExplaratoryDataAnalysis():
    """ """
    
    def __init__(self, dataset, show_plots = False):
        self.dataset = dataset
        self.show_plots = show_plots

    @classmethod
    def get_class_name(cls):
            return cls.__name__


    def _get_null_columns(self):
        return self.dataset.isnull().sum()


    def _get_dataset_head(self, n=5):
        return self.dataset.head(n)


    def _get_dataset_info(self):
        return self.dataset.info()


    def _get_dataset_description(self, numerics_only=False):
        if numerics_only:
            return self.dataset.describe()
        else:
            return self.dataset.describe(include='all')


    def _get_columns_with_nan(self):
        missing_data = self.dataset.isna().sum()
        return missing_data[missing_data > 0]


    def _create_html_raport(self, title):
        path = os.path.join(REPORT_PATH, title)
        profile = ProfileReport(self.dataset, title=title,
                        explorative=True, correlations = {
                        "pearson": {"calculate": True},
                        "spearman": {"calculate": True},
                        "kendall": {"calculate": True}})
        
        common.check_if_path_exists(folder_path=REPORT_PATH)
        profile.to_file(path)


    def _check_amounts_distribution(self, folder_name="BeforeFeatureEngineering"):
        amount = self.dataset['amount'].astype(float)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1 = axes[0]
        ax1.hist(amount, bins=60)
        ax1.set_title('Amount Distribution', fontsize=14)
        ax1.set_xlabel('Amount')
        ax1.set_ylabel('Frequency')
        ax1.grid(axis='y', alpha=0.3)

        log_amount = np.log1p(amount)
        ax2 = axes[1]
        ax2.hist(log_amount, bins=60)
        ax2.set_title('Log(Amount) Distribution', fontsize=14)
        ax2.set_xlabel('log(1 + amount)')
        ax2.set_ylabel('Frequency')
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        common.save_plot(folder_path=f'plots/{self.get_class_name()}/{folder_name}', file_name='LogAmountDistribution.png')


    def _show_unique_values(self):
        for column in self.dataset.columns:
            unique_values = self.dataset[column].unique()
            logger.info(f"Column '{column}' has {len(unique_values)} unique values: {unique_values}")


    def _show_colleration_matrix(self, folder_name="BeforeFeatureEngineering"):
        df_numeric_only = self.dataset.select_dtypes(include=['number'])
        correlation_matrix = df_numeric_only.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5 , fmt=".2f")
        plt.title('Correlation Heatmap of Numerics Obbjects')
        common.save_plot(folder_path=f'plots/{self.get_class_name()}/{folder_name}', file_name='correlation_heatmap.png')


    def _check_fraud_amount(self, folder_name="BeforeFeatureEngineering"):
        fraud_counts = self.dataset['is_fraud'].value_counts()
        fraud_percentage = self.dataset['is_fraud'].value_counts(normalize=True) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.countplot(data=self.dataset, x='is_fraud', ax=ax1)
        ax1.set_title('Amount of Fraud and Non-Fraud Transactions')
        ax1.set_xlabel('Is Fraud')
        ax1.set_ylabel('Count')
        ax1.grid(axis='both', alpha=0.3)
        
        for p in ax1.patches:
            height = p.get_height()
            ax1.text(p.get_x() + p.get_width()/2., height,
                    f'{int(height)}',
                    ha="center", va="bottom", fontsize=10)
        
        bars = ax2.bar(range(len(fraud_percentage)), fraud_percentage.values, color=['green', 'red'])
        ax2.set_title('Fraud vs Non-Fraud Transactions (Percentage) ')
        ax2.set_xlabel('Is Fraud')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_xticks(range(len(fraud_percentage)))
        ax2.set_xticklabels(['Non-Fraud', 'Fraud'], rotation=0)
        ax2.grid(axis='both', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha="center", va="bottom", fontsize=10)
        
        plt.tight_layout()
        common.save_plot(folder_path=f'plots/{self.get_class_name()}/{folder_name}', file_name='is_fraud_analysis.png')
        
        logger.info("\nFraud counts:")
        pprint(fraud_counts)

        logger.info("\nFraud percentage:")
        pprint(fraud_percentage)
    
    def view_into_data(self):
        logger.info("\nFirst 5 rows of dataset:")
        pprint(self._get_dataset_head())
        
        logger.info("\nDataset info:")
        pprint(self._get_dataset_info())

        logger.info("\nDataset description for all columns:")
        pprint(self._get_dataset_description())

        logger.info("\nDataset description for all numeric columns:")
        pprint(self._get_dataset_description(numerics_only=True))

        logger.info("\nNull values in each column:")
        pprint(self._get_null_columns())

        logger.info("\nColumns with NaN values:")
        pprint(self._get_columns_with_nan())

        logger.info("\nUnique values in each column:")
        self._show_unique_values()

    def create_report(self, title, generate_html_report=False):
        if generate_html_report:
            logger.info("\n Generate HTML raport...")
            self._create_html_raport(title=title)
            logger.info(f"Report saved to the {REPORT_PATH +'/'+ title}")
        else :
            logger.warning("\n Skipping HTML raport generation.")


    def get_plots(self, folder_name="BeforeFeatureEngineering"):

        logger.info("\nFraud amount analysis:")
        self._check_fraud_amount(folder_name)

        logger.info("\nAmounts distribution analysis:")
        self._check_amounts_distribution(folder_name)

        logger.info("\nColleration matrix:")
        self._show_colleration_matrix(folder_name)

        if self.show_plots:
            plt.show()

    def get_dataset(self):
        return self.dataset
