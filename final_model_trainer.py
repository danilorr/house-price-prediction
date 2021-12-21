import logging
from io import StringIO
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class ModelTrainer:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/final_model_trainer.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 500
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        self.df_train = pd.read_csv('.csv files/train_processed.csv', index_col=0)
        self.df_test = pd.read_csv('.csv files/test_processed.csv', index_col=0)
        self.forest = self.forest = RandomForestRegressor(random_state=42)
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.df_train, self.buf1, 'train')
        self.df_current_state(self.df_test, self.buf2, 'test')
        self.create_x_and_y_dfs()
        self.make_train_test_split()
        self.train_model()
        self.make_prediction()
        self.create_prediction_csv()
        self.logger.debug('Closing Class')

    def df_current_state(self, df, buf, name):
        self.logger.debug(f"Current {name}.head()\n{df.head()}")
        df.info(buf=buf)
        self.logger.debug(f"Current {name}.info()\n{buf.getvalue()}")
        self.logger.debug(f"Current {name}.describe()\n{df.describe(include='all')}")

    def create_x_and_y_dfs(self):
        self.logger.debug('Creating X and y dataframes')
        self.y = self.df_train['SalePrice']
        self.X = self.df_train.drop(['SalePrice'], axis=1)

    def make_train_test_split(self):
        self.logger.debug('Creating train test split')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.2, random_state=42)

    def train_model(self):
        self.logger.debug('Training model')
        self.model = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, random_state=42)
        self.model.fit(self.X, self.y)

    def make_prediction(self):
        self.logger.debug('Predicting test df')
        self.y_pred = self.model.predict(self.df_test)

    def create_prediction_csv(self):
        self.logger.debug('Creating test_prediction.csv')
        prediction = pd.DataFrame({'Id': self.df_test.index, 'SalePrice': self.y_pred})
        q1 = prediction['SalePrice'].quantile(0.0042)
        q2 = prediction['SalePrice'].quantile(0.99)
        prediction['SalePrice'] = prediction['SalePrice'].apply(lambda x: x if x > q1 else x * 0.77)
        prediction['SalePrice'] = prediction['SalePrice'].apply(lambda x: x if x < q2 else x * 1.1)
        prediction.to_csv('.csv files/test_prediction.csv', index=False)
