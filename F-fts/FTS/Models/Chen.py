import pandas as pd
import numpy as np
from FTS.Evaluate.Validation import validation


class Chen1996:

    def __init__(self, ):
        '''
        Chen model 1996
        Forecasting enrollments based on fuzzy time series
        '''
        self.name_model = 'Chen 1996 model'
        print(self.name_model)


    def __set_universe_discourse(self):
        '''
        U = [Dmin - D1, Dmax + D2]
        '''
        if self.D1 and self.D2:
            U = [np.min(self.ts_array) - self.D1, np.max(self.ts_array) + self.D2]
        else:
            U = [np.min(self.ts_array), np.max(self.ts_array)]
        return U


    def __define_support(self, ):
        '''
        return: Universe of discourse
        '''
        return (abs(self.U[1] - self.U[0]) / self.amount_sets)


    def define_intervals(self, ):

        intervals = []
        midpoints = []

        for d in range(1, self.amount_sets + 1):
            interval = [self.U[0] + (d - 1) * self.support, self.U[0] + d * self.support]
            intervals.append(interval)
            midpoints.append(0.5 * np.sum(interval))

        return intervals, midpoints


    def create_fuzzy_sets(self, name='A.'):

        intervals, midpoints = self.define_intervals()
        instances = []

        self.md = {}

        for i, value in enumerate(midpoints):
            term = name + str(i + 1)
            a = min(intervals[i])
            b = max(intervals[i])
            m = value
            instances.append({'term': term, 'a': a, 'm': m, 'b': b})

            self.md[term] = m

        self.df_mf = pd.DataFrame(instances)


    def __create_fuzzy_sets(self):
        ''' Modeling Fuzzy Sets '''
        self.U = self.__set_universe_discourse()  # ok
        self.support = self.__define_support()
        self.fuzzy_sets = self.create_fuzzy_sets()


    def pertinence(self, x, a, m, b):
        '''
        Calculate pertinence of value x with triangular function
        '''
        if x >= a and x < m:
            return (x - a) / (m - a)
        elif x >= m and x < b:
            return (b - x) / (b - m)
        return 0


    def fuzzify_by_intervals(self, value):
        '''
        obtain the highest degree of relevance of a crips value
        '''
        for i in range(self.df_mf.shape[0]):

            if self.df_mf['a'].iloc[i] <= value <= self.df_mf['b'].iloc[i]:
                return self.df_mf['term'].iloc[i]


    def fuzzify_sets(self, value, only_term=True):
        '''
        Get the highest degree of relevance of a crips value
        '''
        df_res = {'term': '404', 'pertinence': 0}

        for i in range(self.df_mf.shape[0]):

            pertinence = self.pertinence(x=value,
                                         a=self.df_mf['a'].iloc[i],
                                         m=self.df_mf['m'].iloc[i],
                                         b=self.df_mf['b'].iloc[i])

            if pertinence > df_res['pertinence']:
                df_res['pertinence'] = pertinence
                df_res['term'] = self.df_mf['term'].iloc[i]

        if only_term:
            return df_res['term']

        return df_res


    def create_ts_terms(self):
        self.ts_terms = [self.fuzzify_by_intervals(self.ts_array[i]) for i in range(len(self.ts_array))]


    def create_rules(self):
        '''
        :param ts_terms: time series with linguistics terms ([A1, A2, ..., AN])
        :return: base of rules
        '''

        self.create_ts_terms()
         
        # creating dictionary to store relationship groups
        flrg = {}
        for termo in set(self.df_mf['term']): flrg[termo] = []

        # checks relationships (Antecedent -> Consequent ) and groups on terms 
        for i in range(self.partition):
            antecedente = self.ts_terms[i]
            consequente = self.ts_terms[i + 1]
            if consequente not in flrg[antecedente]: flrg[antecedente].append(consequente)

        return flrg


    def MidPoint(self, ):
        '''
        Defuzzification with Midpoint create by [1]
        next terms can be recognized - forecasting only value
        :param flrg: rules base (A.1:[terms], A.2:[terms], ..., A.n:[terms]) lista encadeada
        :param mf(membership function): dict with terms of sets and yours midpoints (A1:midpoint,..., An:midpoint)
        :return: y_pred - vector of numerbs with prediction
        '''

        y_pred = []
        for i in range(self.partition):
            antecedente = self.ts_terms[i]
            y_pred_value = 0
            if len(self.flrg[antecedente]) == 0:
                y_pred_value = self.md[antecedente]
            else:
                pred = [self.md[consequente] for consequente in self.flrg[antecedente]]
                y_pred_value = np.mean(pred)
            y_pred.append(y_pred_value)

        return y_pred




    def MidPointInTest(self,):
        '''
        Defuzzification with Midpoint based by [1]
        forecasting term ans value
        :return: y_pred - vector of numerbs with prediction
        '''
        y_pred = []
        antecedente = self.ts_terms[self.partition]

        for i in range(len(self.ts_terms) - self.partition):

            y_pred_value = 0

            if len(self.flrg[antecedente]) == 0: 
                y_pred_value = self.md[antecedente]

            else:
                pred = [self.md[consequente] for consequente in self.flrg[antecedente]]
                y_pred_value = np.mean(pred)

            y_pred.append(y_pred_value)

            antecedente = self.fuzzify_by_intervals(y_pred_value)

        return y_pred


    def MidPointAuto(self, actual_value, forecasting_size=10):

        y_pred = []

        antecedente = self.fuzzify_by_intervals(actual_value)

        for i in range(forecasting_size):

            y_pred_value = 0

            if len(self.flrg[antecedente]) == 0:
                y_pred_value = self.md[antecedente]

            else:
                pred = [self.md[consequente] for consequente in self.flrg[antecedente]]
                y_pred_value = np.mean(pred)

            y_pred.append(y_pred_value)

            antecedente = self.fuzzify_by_intervals(y_pred_value)

        return y_pred


    def fit(self, ts_array, sets=7, d1=None, d2=None, train=1):
        '''
        make Fuzzyfication and Fuzzy Logical Relationship

        ts: univariate time series
        sets = amount of fuzzy sets (N(size), DIM(dimension=1)
        d1 = increase the beginning of the universe of discourse
        d2 = increase the end of the universe of discourse
        '''
        self.ts_array = ts_array
        self.amount_sets = sets
        self.ts_terms = []
        self.D1 = d1
        self.D2 = d2
        if train == 1:
            self.partition = len(self.ts_array) - 1
        else:
            self.partition = int(len(self.ts_array) * train)
        self.__create_fuzzy_sets()
        self.flrg = self.create_rules()


    def summary(self):
        print('\n','=' * 30)
        print('SUMMARY')
        print('-'*30)
        print('Partition:{} size:{}'.format(self.partition, len(self.ts_array)))
        print('=' * 30)
        print()


    def predict(self, plot=True, SM=True):

        if self.partition == (len(self.ts_array)-1) or self.partition is len(self.ts_array):
            y_true = self.ts_array[1:]
            y_pred = self.MidPoint()

            mape, mae, mse, rmse, dtw = validation().validation_forecasting(_y_true=y_true,
                                                                       _y_pred=y_pred, plot=plot, decimal=2, show_metrics=SM)
            return mape, mae, mse, rmse, dtw
        else:
            y_true = self.ts_array[self.partition:]
            y_pred = self.MidPointInTest()

            val = validation()

            mape, mae, mse, rmse, dtw = val.validation_forecasting(_y_true=y_true,
                                                                       _y_pred=y_pred, plot=False, decimal=2, show_metrics=SM)
            if plot:
                val.plot_forecasting_and_ts(self.ts_array, y_pred, self.partition, np.arange(self.partition, len(self.ts_array)))

            return mape, mae, mse, rmse, dtw



''' ========================================================================== '''
'''                                   EXAMPLES                                 

def __example_Taiex(path_ts, col='avg', year=None):
    # read ts
    ts = pd.read_csv(path_ts, index_col=[0], parse_dates=[0])

    if year:
        ts = pd.DataFrame(ts[col][year])
        print(ts.head(2))
        print('=' * 30)
        print(ts.tail(2))

    # instancie model
    model = Chen1996()

    model.fit(ts=pd.DataFrame(ts[col]), col=col)

    model.forecasting(validation=True)


def __example_Alabama():
    data = pd.read_csv('../data/csv/Enrollments.csv', sep=';', index_col=[0])

    # define model
    model = Chen1996()

    # fitting
    model.fit(ts=data, sets=7, d1=55, d2=663)

    # Forecasting
    model.forecasting(validation=True)


def __example_Temperatura():
    data = pd.read_csv('../data/csv/a1_temperatura.csv', index_col=[0])

    colunas = data.columns.to_list()
    coluna = colunas[0]
    print('Coluna da observação:', coluna)

    model = Chen1996()

    model.fit(ts=pd.DataFrame(data[coluna]), col=coluna)

    model.forecasting(validation=True)


def __example_wang_2013():
    data = [[13000, 13324],
            [13324, 13648],
            [13648, 15078],
            [15078, 15765],
            [15765, 17505],
            [17505, 18902],
            [18902, 20000]]

    model = Chen1996()

    model.fit(ts=data)

    model.forecasting(validation=True)
    
========================================================================== '''
