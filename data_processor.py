import logging
from io import StringIO
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataProcessor:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/data_processor.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 500
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        self.df_train = pd.read_csv('.csv files/train.csv')
        self.df_test = pd.read_csv('.csv files/test.csv')
        self.combine = pd.concat((self.df_train.loc[:, 'Id':'SaleCondition'],
                                  self.df_test.loc[:, 'Id':'SaleCondition']))
        self.buf1 = StringIO()
        self.buf2 = StringIO()
        self.le = LabelEncoder()

    def start(self):
        self.logger.debug('Starting Class')
        self.combine_current_state(self.buf1)
        self.numeric_feature_evaluator()
        self.ordinal_feature_evaluator()
        self.fix_outliers()
        self.extra_drops()
        self.fill_features_na()
        self.feature_encoder()
        self.combine_current_state(self.buf2)
        self.update_train_and_test_dfs()
        self.create_train_df_csv()
        self.create_test_df_csv()
        self.logger.debug('Closing Class')

    def combine_current_state(self, buf):
        self.logger.debug(f"Current combine.head()\n{self.combine.head()}")
        self.combine.info(buf=buf)
        self.logger.debug(f"Current combine.info()\n{buf.getvalue()}")
        self.logger.debug(f"Current combine.describe()\n{self.combine.describe(include='all')}")

    def check_feature_values(self, feature):
        self.logger.debug(f"{feature}'s values:\nValue counts:\n{self.df_train[feature].value_counts()}"
                          f"\nTotal Nulls: {self.df_train[feature].isna().sum()}")

    def check_features_linear_correlation(self, feature1, feature2):
        self.logger.debug(f'Checking linear correlation between {feature1} and {feature2}:\n'
                          f"Correlation: {self.df_train[feature1].corr(self.df_train[feature2])}")

    def drop_feature(self, feature):
        self.logger.debug(f'Dropping feature {feature} from dataframe')
        self.combine = self.combine.drop(columns=feature)

    def numeric_feature_evaluator(self):
        self.check_feature_values('Street')
        self.logger.debug(f"99.6% of the values of the column Street is Pave, so the column isn't going to be used")
        self.drop_feature('Street')

        self.check_feature_values('Alley')
        self.logger.debug(f"94% of the values of the column Alley are NaN, so the column isn't going to be used")
        self.drop_feature('Alley')

        self.check_feature_values('LotShape')
        self.logger.debug(f"Only 3% of the values fall under IR2 or IR3. Maybe only Reg and IR1 will be considered")

        self.check_feature_values('LandContour')
        self.logger.debug(f"90% of the values fall under Lvl. The column will be dropped")
        self.drop_feature('LandContour')

        self.check_feature_values('Utilities')
        self.logger.debug(f"Only one value was marked as NoSeWa. It's the element if Id 945. The row 944 may be "
                          f"discarded for being a relevant outlier as well as the column Utilities")
        self.drop_feature('Utilities')

        self.check_feature_values('LotConfig')

        self.check_feature_values('LandSlope')
        self.logger.debug(f"95% of the rows fall under the Gtl value. The column will be discarded")
        self.drop_feature('LandSlope')

        self.check_feature_values('Neighborhood')
        self.logger.debug(f"Too many possible values for Neighborhood. The feature can be remade into "
                          f"smaller categories")

        self.check_feature_values('Condition1')
        self.check_feature_values('Condition2')
        self.logger.debug(f"Condition1 values will be grouped into Norm/Anorm and Condition2 Feature will "
                          f"be discarded (99% of the values fall under 1 category)")
        self.drop_feature('Condition2')

        self.check_feature_values('HouseStyle')

        self.check_features_linear_correlation('YearBuilt', 'YearRemodAdd')
        self.check_features_linear_correlation('YearBuilt', 'SalePrice')
        self.check_features_linear_correlation('YearRemodAdd', 'SalePrice')
        self.logger.debug(f"Naturally YearBuilt and YearRemodAdd have high correlation. Between both, YearBuilt "
                          f"has a slightly higher correlation with SalesPrice. So for now we'll use it")
        self.drop_feature('YearRemodAdd')

        self.check_feature_values('RoofMatl')
        self.logger.debug(f"98% of the values fall under CompShg. The column will be discarded")
        self.drop_feature('RoofMatl')

        self.check_feature_values('Exterior1st')
        self.logger.debug(f"Too many categories. New categories should be created to contain the previous ones")

        self.check_feature_values('Exterior2nd')
        self.logger.debug(f"Too many categories. New categories should be created to contain the previous ones")

        self.check_feature_values('MasVnrType')
        self.logger.debug(f"MasVnrType seem to have a good correlation with SalePrice. "
                          f"MasVnrArea is highly correlated to MasVnrType and should be merged or dropped")
        self.drop_feature('MasVnrArea')

        self.check_feature_values('BsmtCond')
        self.logger.debug(f"90% of the values fall under TA. The column should be dropped")
        self.drop_feature('BsmtCond')

        self.check_feature_values('BsmtFinType1')
        self.logger.debug(f"BsmtFinSF1 and BsmtFinType1 has high correlation")

        self.check_feature_values('BsmtFinType2')
        self.logger.debug(f"87% of the values are Unfinished. The column will be discarded, together with BsmtFinSF2")
        self.drop_feature('BsmtFinType2')
        self.drop_feature('BsmtFinSF2')

        self.check_features_linear_correlation('BsmtUnfSF', 'SalePrice')
        self.logger.debug(f"The column BsmtUnfSF has too low correlation with SalePrice. Will be dropped")
        self.drop_feature('BsmtUnfSF')

        self.check_feature_values('Heating')
        self.logger.debug(f"98% of the values fall under GasA. Column will be discarded")
        self.drop_feature('Heating')

        self.check_feature_values('CentralAir')
        self.logger.debug(f"The correlation between SalePrice and CentralAir isn't enough to justify keeping "
                          f"the feature since it has 94% of the rows with one value.")
        self.drop_feature('CentralAir')

        self.check_feature_values('LowQualFinSF')
        self.logger.debug(f"98% of the values of LowQualFinSF is 0. Therefore, the column will de dropped")
        self.drop_feature('LowQualFinSF')

        self.check_feature_values('BsmtFullBath')
        self.check_feature_values('HalfBath')
        self.logger.debug(f"Maybe BsmtFullBath, BsmtHalfBath, FullBath and HalfBath could become one single feature")

        self.check_features_linear_correlation('BedroomAbvGr', 'SalePrice')
        self.logger.debug(f"BedroomAbvGr column has low correlation with SalePrice. Will be dropped")
        self.drop_feature('BedroomAbvGr')

        self.check_features_linear_correlation('KitchenAbvGr', 'SalePrice')
        self.logger.debug(f"KitchenAbvGr column has low correlation with SalePrice. Will be dropped")
        self.drop_feature('KitchenAbvGr')

        self.check_feature_values('Functional')
        self.logger.debug(f"93% of the values fall under Typ, so the column will de dropped . "
                          f"Also, rows 398, 406, 662, 666, 710 and 1013 should be considered outliers and dropped")
        self.drop_feature('Functional')

        self.check_feature_values('TotRmsAbvGrd')

        self.check_feature_values('Fireplaces')
        self.check_feature_values('FireplaceQu')
        self.logger.debug(f"Both columns maybe should become a single feature")

        self.check_feature_values('GarageType')
        self.check_features_linear_correlation('GarageCars', 'GarageArea')
        self.logger.debug(f"98% of the values of LowQualFinSF is 0. Therefore, the column will de dropped")

        self.check_feature_values('PavedDrive')
        self.logger.debug(f"92% of the values fall under Y, so the column should be dropped")
        self.drop_feature('PavedDrive')

        self.check_features_linear_correlation('WoodDeckSF', 'OpenPorchSF')
        self.check_features_linear_correlation('WoodDeckSF', 'SalePrice')
        self.check_features_linear_correlation('OpenPorchSF', 'SalePrice')
        self.logger.debug(f"Correlation between both features isn't enough to build a new feature. "
                          f"Correlation between the 2 features and SalePrice isn't strong enough to keep them")
        self.drop_feature('WoodDeckSF')
        self.drop_feature('OpenPorchSF')

        self.check_feature_values('EnclosedPorch')
        self.check_feature_values('3SsnPorch')
        self.check_feature_values('ScreenPorch')
        self.logger.debug(f"All the 3 features have too high number of zeroes and will be discarded")
        self.drop_feature('EnclosedPorch')
        self.drop_feature('3SsnPorch')
        self.drop_feature('ScreenPorch')

        self.check_feature_values('PoolQC')
        self.logger.debug(f"99.5% of the values are empty. The column will be dropped, "
                          f"together with their dependent column PoolArea")
        self.drop_feature('PoolQC')
        self.drop_feature('PoolArea')

        self.check_feature_values('MiscFeature')
        self.logger.debug(f"96% of the values are NaN. The column will be dropped, "
                          f"together with their dependent column MiscVal")
        self.drop_feature('MiscFeature')
        self.drop_feature('MiscVal')

        self.check_features_linear_correlation('MoSold', 'SalePrice')
        self.check_features_linear_correlation('YrSold', 'SalePrice')
        self.logger.debug(f"Neither MoSold nor YrSold has any correlation with SalePrice, so both will be dropped")
        self.drop_feature('MoSold')
        self.drop_feature('YrSold')

        self.check_feature_values('SaleType')
        self.logger.debug(f"Too many categories. Maybe a new category could be created")

        self.check_feature_values('SaleCondition')
        self.logger.debug(f"Too many categories. Maybe a new category could be created")

        self.logger.debug(f"Dropping other numerical features due to low correlation with target feature")
        drop_col = ['LotFrontage', 'LotArea', 'OverallCond', 'BsmtFinSF1', '2ndFlrSF',
                    'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'Fireplaces', 'GarageCars']
        self.drop_feature(drop_col)

    def transform_cat_to_ordinal(self, feature, category, ordinal):
        self.df_train[feature] = self.df_train[feature].replace(category, ordinal)

    def ordinal_feature_evaluator(self):
        self.transform_cat_to_ordinal('LotShape', ['IR3', 'IR2', 'IR1', 'Reg'], [1, 2, 3, 4])
        self.check_features_linear_correlation('LotShape', 'SalePrice')

        self.transform_cat_to_ordinal('LotConfig', ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], [1, 2, 3, 4, 5])
        self.check_features_linear_correlation('LotConfig', 'SalePrice')

        self.transform_cat_to_ordinal('ExterQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5])
        self.check_features_linear_correlation('ExterQual', 'SalePrice')

        self.transform_cat_to_ordinal('ExterCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5])
        self.check_features_linear_correlation('ExterCond', 'SalePrice')

        self.transform_cat_to_ordinal('BsmtQual', ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])
        self.check_features_linear_correlation('BsmtQual', 'SalePrice')

        self.transform_cat_to_ordinal('BsmtExposure', ['NA', 'No', 'Mn', 'Av', 'Gd'], [0, 1, 2, 3, 4])
        self.check_features_linear_correlation('BsmtExposure', 'SalePrice')

        self.transform_cat_to_ordinal('BsmtFinType1', ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
                                      [0, 1, 2, 3, 4, 5, 6])
        self.check_features_linear_correlation('BsmtFinType1', 'SalePrice')

        self.transform_cat_to_ordinal('HeatingQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5])
        self.check_features_linear_correlation('HeatingQC', 'SalePrice')

        self.transform_cat_to_ordinal('KitchenQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5])
        self.check_features_linear_correlation('KitchenQual', 'SalePrice')

        self.transform_cat_to_ordinal('FireplaceQu', ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])
        self.check_features_linear_correlation('FireplaceQu', 'SalePrice')

        self.transform_cat_to_ordinal('GarageFinish', ['NA', 'Unf', 'RFn', 'Fin'], [0, 1, 2, 3])
        self.check_features_linear_correlation('GarageFinish', 'SalePrice')

        self.transform_cat_to_ordinal('GarageQual', ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])
        self.check_features_linear_correlation('GarageQual', 'SalePrice')

        self.transform_cat_to_ordinal('GarageCond', ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])
        self.check_features_linear_correlation('GarageCond', 'SalePrice')

        self.logger.debug(f"Based on their correlation with SalePrice, some columns will be dropped")
        self.drop_feature(['LotShape', 'LotConfig', 'ExterCond', 'BsmtExposure',
                           'BsmtFinType1', 'HeatingQC', 'GarageQual', 'GarageCond'])

    def fix_outliers(self):
        self.logger.debug('Removing outliers based on GrLivArea')
        self.combine = self.combine.drop(self.combine[self.combine['Id'] == 1299].index)
        self.combine = self.combine.drop(self.combine[self.combine['Id'] == 524].index)

    def extra_drops(self):
        # Other columns were dropped based on:
        # Neighborhood (too many categories), BldgType (correlation with HouseStyle), RoofStyle (too many of one category)
        # Exterior2nd (correlation with exterior1st), Electrical (too many of one category), FireplaceQu (correlation with Fireplaces),
        # GarageType (correlation with GarageArea), GarageYrBlt (correlation with GaregeArea), GarageFinish (correlation with GarageArea),
        # Fence (too many of one category), SaleType (too many of one category), SaleCondition (too many of one category)
        self.logger.debug('Dropping other columns based on empirical evidence')
        self.drop_feature(['BldgType', 'RoofStyle', 'Exterior2nd', 'Electrical', 'FireplaceQu', 'GarageType',
                           'GarageYrBlt', 'GarageFinish', 'Fence', 'SaleType', 'SaleCondition', 'Id'])

    def fill_na(self, feature, value):
        self.combine[feature] = self.combine[feature].fillna(value)

    def fill_features_na(self):
        self.logger.debug('Filling all nulls with the adequate value')
        self.fill_na('BsmtQual', 'None')
        self.fill_na('MasVnrType', 'None')
        self.fill_na('MSZoning', 'RL')
        self.fill_na('GarageArea', 0)
        self.fill_na('KitchenQual', 'TA')
        self.fill_na('Exterior1st', 'Other')
        self.fill_na('TotalBsmtSF', 0)
        self.logger.debug(f"Checking if all null values were filled:\n"
                          f"{self.combine.isnull().sum().sort_values(ascending=False)}")

    def feature_encoder_base(self, feat_list):
        self.logger.debug('Encoding object features')
        for feat in feat_list:
            self.combine[feat] = self.le.fit_transform(self.combine[feat])

    def feature_encoder(self):
        self.feature_encoder_base(['MSZoning', 'Neighborhood', 'Condition1', 'HouseStyle', 'Exterior1st',
                                   'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'KitchenQual'])

    def update_train_and_test_dfs(self):
        self.logger.debug('Creating X and y dataframes')
        self.y_train = self.df_train['SalePrice']
        self.X_train = self.combine[:self.df_train.shape[0]]
        self.df_train = self.X_train.join(self.y_train)

        self.df_test = self.combine[self.df_train.shape[0]:]
        self.df_test.index += 1461

    def create_train_df_csv(self):
        self.logger.debug('Creating train.csv')
        self.df_train.to_csv(r'.csv files/train_processed.csv')

    def create_test_df_csv(self):
        self.logger.debug('Creating test.csv')
        self.df_test.to_csv(r'.csv files/test_processed.csv')
