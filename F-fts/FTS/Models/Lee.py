import pandas as pd
import numpy as np
from FTS.Evaluate.Validation import validation


class Lee2009:

    def __init__(self,):
        self.name_description = 'Weighted rules'

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
        '''
        Modeling Fuzzy Sets
        '''

        self.U = self.__set_universe_discourse()  # ok
        self.support = self.__define_support()
        self.fuzzy_sets = self.create_fuzzy_sets()


    def pertinence(self, x, a, m, b):
        '''
        calculate membership of value x with triangular function
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


    def create_rules_LEE(self):
        '''
        weigthed rules
        :param ts_terms: time series with linguistics terms ([A1, A2, ..., AN])
        :return: base of rules
        '''
        self.create_ts_terms()
        #creating dictionary to store relationship groups

        flrg = {}
        for termo in set(self.df_mf['term']): flrg[termo] = []

        for i in range(self.partition):
            antecedente = self.ts_terms[i]
            consequente = self.ts_terms[i + 1]
            flrg[antecedente].append(consequente)

        return flrg

    def defuzzification_LEE(self, consequentes):
        """
        Step 8: Assigning weights.
        consequentes: list with consequent terms
        return: defuzzified value
        """

        weigths = np.arange(1, (len(consequentes) + 1), dtype=np.float32)
        weigths /= np.sum(weigths)

        mid_points = []
        for term_cons in consequentes: mid_points.append(float(self.md[term_cons]))
        cons_midpoints = np.array(mid_points)

        return sum(weigths * cons_midpoints.T)


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

            y_pred.append(self.defuzzification_LEE(self.flrg[antecedente]))

        return y_pred


    def MidPointInTest(self, ):
        '''
        Defuzzification with Midpoint based by [1]
        forecasting term ans value
        :return: y_pred - vector of numerbs with prediction
        '''
        y_pred = []
        antecedente = self.ts_terms[self.partition]
        for i in range(len(self.ts_terms) - self.partition):

            if len(self.flrg[antecedente]) == 0: # verify if antecedent not contais consequent
                y_pred_value = self.defuzzification_LEE([antecedente])
            else:
                y_pred_value = self.defuzzification_LEE(self.flrg[antecedente])

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
        self.flrg = self.create_rules_LEE()


    def summary(self):
        print('\n', '=' * 30)
        print('SUMMARY')
        print('-' * 30)
        print('FLRG:\n', self.flrg)
        print('=' * 30)
        print()


    def predict(self, plot=True, SM=True):
        if self.partition  == (len(self.ts_array) - 1):
            y_true = self.ts_array[1:]
            y_pred = self.MidPoint()

            self.y_pred = y_pred
            self.y_true = y_true

            # visualize
            mape, mae, mse, rmse, dtw = validation().validation_forecasting(_y_true=y_true.copy(),
                                                                       _y_pred=y_pred.copy(), plot=plot, decimal=2, show_metrics=SM)
            return mape, mae, mse, rmse, dtw

        # forecasting in test partition
        else:
            y_true = self.ts_array[self.partition:]
            y_pred = self.MidPointInTest()

            val = validation()

            mape, mae, mse, rmse, dtw = val.validation_forecasting(_y_true=y_true,
                                                              _y_pred=y_pred, plot=False, decimal=2, show_metrics=SM)
            if plot:
                val.plot_forecasting_and_ts(self.ts_array, y_pred, self.partition,
                                        np.arange(self.partition, len(self.ts_array)))

            return mape, mae, mse, rmse, dtw