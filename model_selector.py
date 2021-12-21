import logging
from io import StringIO
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso


class ModelSelector:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/model_selector.log', mode='w')
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
        self.forest = RandomForestRegressor(random_state=42)
        self.linear = LinearRegression()
        self.ridge = Ridge()
        self.lasso = Lasso()
        self.mods = [self.forest, self.linear, self.ridge, self.lasso]
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.df_train, self.buf1, 'train')
        self.df_current_state(self.df_test, self.buf2, 'test')
        self.create_x_and_y_dfs()
        self.make_train_test_split()
        self.model_tester()
        self.grind_searcher()
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

    def model_tester(self):
        for m in self.mods:
            model = m
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            self.logger.debug(f" MSE from {model}: {mean_squared_error(self.y_test, y_pred, squared=False)}")
        self.logger.debug(f"The model with best result was Random Forest")

    def grind_searcher(self):
        model = RandomForestRegressor()

        param = {
            'n_estimators': (50, 100, 1000),
            'min_samples_leaf': (2, 3)
        }

        grid = GridSearchCV(model, param)
        grid.fit(self.X_train, self.y_train)

        self.logger.debug(f'\nThe best parameters are:\n{grid.best_params_}')
        self.logger.debug(f'The best score is: {grid.best_score_}')
