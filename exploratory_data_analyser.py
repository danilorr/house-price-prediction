import logging
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class DataAnalyser:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/exploratory_data_analyser.log', mode='w')
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
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.buf1)
        self.sales_dist_plot()
        self.lot_regplot()
        self.year_qual_pairplot()
        self.vnrtype_boxplot()
        self.centralair_swarmplot()
        self.garage_regplot()
        self.grlivarea_outlier_scatterplot()

    def df_current_state(self, buf):
        self.logger.debug(f"Current combine.head()\n{self.df_train.head()}")
        self.df_train.info(buf=buf)
        self.logger.debug(f"Current combine.info()\n{buf.getvalue()}")
        self.logger.debug(f"Current combine.describe()\n{self.df_train.describe(include='all')}")

    def sales_dist_plot(self):
        self.logger.debug(f'Generating sales_dist_plot.png')
        fig, ax = plt.subplots()
        sns.distplot(self.df_train['SalePrice'], ax=ax)
        plt.savefig('plots/sales_dist_plot.png')

    def lot_regplot(self):
        self.logger.debug(f'Generating lot_regplot.png')
        fig, ax = plt.subplots()
        sns.regplot(self.df_train['LotFrontage'], self.df_train['LotArea'], ax=ax)
        plt.savefig('plots/lot_regplot.png')

    def year_qual_pairplot(self):
        self.logger.debug(f'Generating year_qual_pairplot.png')
        g = sns.PairGrid(self.df_train, vars=["YearBuilt", "OverallQual", "OverallCond"])
        g.map(sns.scatterplot)
        plt.savefig('plots/year_qual_pairplot.png')

    def vnrtype_boxplot(self):
        self.logger.debug(f'Generating vnrtype_boxplot.png')
        fig, ax = plt.subplots()
        sns.boxplot(x='MasVnrType', y='SalePrice', data=self.df_train, ax=ax)
        plt.savefig('plots/vnrtype_boxplot.png')

    def centralair_swarmplot(self):
        self.logger.debug(f'Generating centralair_swarmplot.png')
        fig, ax = plt.subplots()
        sns.swarmplot(x='CentralAir', y='SalePrice', data=self.df_train, ax=ax)
        plt.savefig('plots/centralair_swarmplot.png')

    def garage_regplot(self):
        self.logger.debug(f'Generating garage_regplot.png')
        fig, ax = plt.subplots()
        sns.regplot(self.df_train['GarageCars'], self.df_train['GarageArea'], ax=ax)
        plt.savefig('plots/garage_regplot.png')

    def grlivarea_outlier_scatterplot(self):
        self.logger.debug(f'Generating grlivarea_outlier_scatterplot.png')
        fig, ax = plt.subplots()
        sns.scatterplot(self.df_train['GrLivArea'], self.df_train['SalePrice'], ax=ax)
        plt.savefig('plots/grlivarea_outlier_scatterplot.png')
