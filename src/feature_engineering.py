#|--------------------------------------------------------------|
#|                          Requirements                        |
#|--------------------------------------------------------------|

import pandas as pd
import numpy as np
import logging

#|--------------------------------------------------------------|
#|                          Maros                               |
#|--------------------------------------------------------------|


#|--------------------------------------------------------------|
#|                          Main part                           |
#|--------------------------------------------------------------|

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self, dataset):
        self.dataset = dataset
        self.unused_columns = []


    ## SECTION 1: TIME FEATURES
    def _set_to_datetime(self):
        self.dataset['transaction_datetime'] = pd.to_datetime(self.dataset['transaction_time'])
        self.unused_columns.append('transaction_datetime')
        self.unused_columns.append('transaction_time')

    def _check_if_night_transaction(self, column_name='transaction_datetime'):
        nightHours = (22, 23, 0, 1, 2, 3, 4, 5)
        self.dataset['is_night_transaction'] = self.dataset[column_name].dt.hour.isin(nightHours).astype(int)

    def _check_if_weekend_transaction(self, column_name='transaction_datetime'):
        weekendDays = (5, 6)
        self.dataset['is_weekend_transaction'] = self.dataset[column_name].dt.dayofweek.isin(weekendDays).astype(int)

    def _check_if_christmast_transaction(self, column_name='transaction_datetime'):
        isDecember = self.dataset[column_name].dt.month.isin([12]).astype(int)
        isNearChristmas = self.dataset[column_name].dt.day.isin(range(15, 27)).astype(int)
        self.dataset['is_christmas_transaction'] = (isDecember & isNearChristmas).astype(int)

    def _datetime_featuring(self):
        self._set_to_datetime()
        self._check_if_night_transaction()
        self._check_if_weekend_transaction()
        self._check_if_christmast_transaction()


    ## SECTION 2: GEO FEATURES
    def _check_if_country_mismatch(self):
        self.dataset['is_country_mismatch'] = (self.dataset['country'] != self.dataset['bin_country']).astype(int)


    ## SECTION 3: USER ACCOUNT FEATURES
    def _add_account_age_features(self):
        self.dataset['is_new_account'] = (self.dataset['account_age_days'] < 30).astype(int)
        self.dataset['is_old_account'] = (self.dataset['account_age_days'] > 365).astype(int)


    ## SECTION 4: AMOUT FEATURES
    def _add_amount_indicators(self):
        self.dataset["user_std_amount"] = self.dataset.groupby('user_id')['amount'].agg(['std'])
        self.dataset['user_std_amount'] = self.dataset['user_std_amount'].fillna(0)
        
        # z-score: (x - μ) / σ
        self.dataset['zscore_amount_user'] = (
            self.dataset['amount'] - self.dataset['avg_amount_user']
        ) / self.dataset['user_std_amount']

        self.dataset['zscore_amount_user'] = self.dataset['zscore_amount_user'].fillna(0, inplace=True)
        
        # TRUE for transactions (> mean_value + 2σ)
        self.dataset['is_high_amount_user'] = self.dataset['amount'] > (self.dataset['avg_amount_user'] + 2 * self.dataset['user_std_amount'])
        self.dataset['is_high_amount_user'].astype(int)


    ## SECTION 5: LOG TRANSFORMATIONS
    def _use_log1p(self, column_name):
        column = self.dataset[column_name].astype(float)
        self.dataset[column_name] = np.log1p(column)
        self.dataset[column_name + "_log1p"] = self.dataset[column_name].replace([np.inf, -np.inf, np.nan], 0)
        self.unused_columns.append(column_name)

    def _run_all_log1p_transforms(self):
        self._use_log1p("user_std_amount")
        self._use_log1p("zscore_amount_user")
        self._use_log1p("shipping_distance_km")
        self._use_log1p("amount")
        self._use_log1p("avg_amount_user")


    ## SECTION 6: ONE-HOT ENCODING
    def _one_hot_encode_column(self, column_name):
        dummies = pd.get_dummies(self.dataset[column_name], prefix=column_name)
        self.dataset = pd.concat([self.dataset, dummies], axis=1)
        self.unused_columns.append(column_name)

    def _run_all_one_hot_encodings(self):
        self._one_hot_encode_column('merchant_category')
        self._one_hot_encode_column('channel')
        self._one_hot_encode_column('country')
        self._one_hot_encode_column('bin_country')


    ## SECTION 7: DATA CLEANING
    def _check_if_column_is_unique(self):
        print(self.dataset.columns)
        for column_name in self.dataset.columns:
            unique_values = set(self.dataset[column_name].unique())
            if unique_values == {0} or unique_values == {1}:
                self.dataset.drop(column_name, axis=1, inplace=True, errors='ignore')
                logger.warning(f"Warning: Feature '{column_name}' dropped due to zero variance.")

    def _delete_unused_columns(self, unused_columns: list):
        unused_columns += self.unused_columns
        # cast to set to avoid duplicates
        for column in set(unused_columns):
            self.dataset.drop(column, axis=1, inplace=True)


    ## SECTION 8: MAIN METHOD TO START  
    def run_feature_engineering(self):
        self._check_if_country_mismatch()
        self._add_account_age_features()
        self._add_amount_indicators()
        self._datetime_featuring()
        
        self._run_all_log1p_transforms()
        self._run_all_one_hot_encodings()
        
        self._check_if_column_is_unique()
        self._delete_unused_columns(['user_id', 'transaction_id'])

        return self.dataset
