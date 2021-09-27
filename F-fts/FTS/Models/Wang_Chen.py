
'''
implementation of paper:
Effective intervals determined by information granules to improve
forecasting in fuzzy time series
'''

from fcmeans import FCM
import numpy as np
import pandas as pd
from FTS.Evaluate.Validation import validation

class Wang2013_Chen:

    def __init__(self):
        self.name = 'Partitioning by FCM and information granules'

    def __set_universe_discourse(self):
        '''
        U = [Dmin - D1, Dmax + D2]
        '''
        if self.D1 and self.D2:
            U = [np.min(self.ts_array) - self.D1, np.max(self.ts_array) + self.D2]
        else:
            U = [np.min(self.ts_array), np.max(self.ts_array)]
        return U


    def __average_dis(self, X):
        '''
        Caculate the average distance of two adjacent data amount set X

        Parameters
        ----------
        X: 1D array
            Universe of discourse
        Return
        ------
        average distance
        '''
        ave = [((X[i + 1] - X[i]) ** 2) for i in range(len(X) - 1)]
        ave_dis = (np.sum(ave) / (len(X) - 1))
        return ave_dis ** 0.5


    def __clustering(self,):

        ''' clustering '''
        fcm = FCM(n_clusters=self.C, m=2)
        fcm.fit(self.ts_array.reshape(self.ts_array.shape[0],1)) # clustering data with shape (22, 1) (N, DIM)

        # outputs
        self.fcm_centers = fcm.centers
        self.u = fcm.u
        self.fcm_labels = fcm.u.argmax(axis=1)


    def __build_prototypes(self):

        self.__clustering()

        self.prototypes = {}
        for i in range(self.C): self.prototypes['v' + str(i + 1)] = []

        for i, value in enumerate(sorted(self.fcm_centers)): self.prototypes['v' + str(i + 1)] = int(value[0])

        # list of prototypes
        self.list_prototypes = list(self.prototypes.values())


    def __build_subsets(self):
        self.md_prototypes = {}

        for i in range(len(self.list_prototypes) - 1):
            self.md_prototypes['m' + str(i + 1)] = int((self.list_prototypes[i] + self.list_prototypes[i + 1]) / 2)

        limites = []
        limites.append(self.Dmin)
        for mid in self.md_prototypes.values(): limites.append(mid)
        limites.append(self.Dmax)


        self.sub_sets = {}
        for i in range(1, self.C + 1): self.sub_sets['D' + str(i)] = []

        for _, x in enumerate(self.ts_array):

            for i in range(len(limites) - 1):
                if limites[i] <= x <= limites[i + 1]:
                    self.sub_sets['D' + str(i + 1)].append(x)


    def __build_granules(self, alpha=0.001):
        '''
        build a information granules
        '''
        Dx = []
        f2 = []
        for indice, i in enumerate(self.sub_sets):

            max_value_a, f2_value_a, max_f2_a = 0, 0, 0
            max_value_b, f2_value_b, max_f2_b = 0, 0, 0

            i_a, i_b = 1, 1
            maior, menor = 0, 1

            for value in self.sub_sets[i]:

                # calculate a and b
                if int(value) <= self.list_prototypes[indice]:
                    f2_value_a = np.exp(-alpha * value) * i_a
                    i_a += 1
                else:
                    f2_value_b = np.exp(-alpha * value) * i_b
                    i_b += 1

                # find [a,b]
                if f2_value_a < menor:
                    menor = f2_value_a
                    max_value_a = value
                    max_f2_a = f2_value_a

                if f2_value_b > maior:
                    maior = f2_value_b
                    max_value_b = value
                    max_f2_b = f2_value_b

            Dx.append([max_value_a, max_value_b])
            f2.append([max_f2_a, max_f2_b])

        return Dx, f2


    def __build_intervals(self,):
        '''
        calculate the difference in distances
        '''
        dis = self.__average_dis(self.ts_array)

        intervals = []

        if (self.K % 2) == 0: 
            #print('IN PAR')
            intervals.append(self.U[0])
            for i in range(self.C-1):
                intervals.append(self.list_prototypes[i]) # add median
                intervals.append(np.mean([self.granules[i][1], self.granules[i+1][0]]))

            intervals.append(self.list_prototypes[-1])
            intervals.append(self.U[1])

        else: # odd
            intervals.append(self.U[0])  # add begin universe of discourse
            if (self.granules[0][0] - dis/2) <= self.Dmin:
                intervals.append((self.U[0] + self.list_prototypes[0])/2)
            else:
                intervals.append((self.granules[0][0] - dis)/2)

            for i in range(self.C-1):
                intervals.append(self.list_prototypes[i]) # add median
                intervals.append(np.mean([self.granules[i][1], self.granules[i+1][0]]))

            intervals.append(self.list_prototypes[-1])
            intervals.append(self.U[1])  # add end universe of discours

        self.support = [[intervals[i], intervals[i+1]] for i in range(self.K)]


    def __create_fuzzy_sets(self, name='A.'):
        self.md = {}
        conjunto = []
        for i in range(self.K):
            term = name + str(i + 1)
            a = min(self.support[i])
            b = max(self.support[i])
            m = np.mean(self.support[i])
            conjunto.append({'term': term, 'a': a, 'm': m, 'b': b})

            self.md[term] = m

        self.df_mf = pd.DataFrame(conjunto)


    def fuzzify_by_intervals(self, value):
        '''
        obtain the highest degree of relevance of a crips value
        '''
        for i in range(self.df_mf.shape[0]):

            if self.df_mf['a'].iloc[i] <= value <= self.df_mf['b'].iloc[i]:
                return self.df_mf['term'].iloc[i]

    def create_ts_terms(self):
        self.ts_terms = [self.fuzzify_by_intervals(self.ts_array[i]) for i in range(len(self.ts_array))]

    def create_rules(self):
        '''
        :param ts_terms: time series with linguistics terms ([A1, A2, ..., AN])
        :return: base of rules
        '''

        self.create_ts_terms()
        '''
        creating dictionary to store relationship groups
        '''

        flrg = {}
        for termo in set(self.df_mf['term']): flrg[termo] = []

        for i in range(self.partition):
            antecedente = self.ts_terms[i]
            consequente = self.ts_terms[i + 1]
            if consequente not in flrg[antecedente]: flrg[antecedente].append(consequente)

        return flrg


    def fit(self, ts_array, sets=7, d1=None, d2=None, train=1):
        '''
       Adjusts the universe of speech based on the granules of information

        Parameters
        ----------
        ts_array: 1D array
            Time series
        sets: int
            numbers of fuzzy sets
        d1, d2: lower and upper limit to add in the universe of discourse
        '''

        self.ts_array = ts_array
        self.K = sets
        self.C = int(self.K / 2) # clustes for clustering FCM
        self.ts_terms = []
        self.D1 = d1
        self.D2 = d2
        self.Dmin = min(ts_array)
        self.Dmax = max(ts_array)
        self.U = self.__set_universe_discourse()

        ''' ======= WUANG MODEL ========='''
        self.__build_prototypes()

        self.__build_subsets()

        self.__build_granules()

        self.granules, _, = self.__build_granules()

        self.__build_intervals()

        #self.summary()
        ''' ============================'''

        # define partition of train and test
        if train == 1:
            self.partition = len(self.ts_array) - 1
        else:
            self.partition = int(len(self.ts_array) * train)

        # create fuzzy set
        self.__create_fuzzy_sets()

        # create rules
        self.flrg = self.create_rules()


    ''' ======== DEFUZZIFICAÇÂO =============='''
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


    def summary(self):
        print('Universe of Discourse:\n', self.U)
        print('Prototypes:\n', self.prototypes)
        print('Prototypes List:\n',self.list_prototypes)
        print('Midpoints of prototypes: \n', self.md_prototypes)
        print('Subsets:\n', self.sub_sets)
        print('Granules of Information:\n', self.granules)
        print('Intervals:\n', self.support)


    def predict(self, plot=True, SM=True):
        if self.partition == (len(self.ts_array)-1):
            y_true = self.ts_array[1:]
            y_pred = self.MidPoint()

            mape, mae, mse, rmse, dtw = validation().validation_forecasting(_y_true=y_true.copy(),
                                                                       _y_pred=y_pred.copy(), plot=plot, points=True, decimal=2, show_metrics=SM)
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

