from fcmeans import FCM
from FTS.Evaluate.Validation import validation
import numpy as np
import pandas as pd


class Auto_MV_FCMB_CHEN:

    def __init__(self):
        ''' ==========================================================
            =============== FTS Forecasting Auto ===================== 
            ========================================================== '''
        print('FTS Forecasting Auto Chen')

    def clustering_FCM(self):

        self.fcm = FCM(n_clusters=self.C, m=2)
        self.fcm.fit(self.X)

        # outputs
        self.fcm_centers = self.fcm.centers
        self.u = self.fcm.u
        self.fcm_labels = self.fcm.u.argmax(axis=1)

        self.col = 'ts'
        self.col_imf = 'imfs'
        self.col_term = 'terms'

        # create membership function
        self.terms = ['A.' + str(i) for i in range(self.C)]

        self.mf = pd.DataFrame(
            {self.col: self.fcm_centers[:, 0], self.col_imf: self.fcm_centers[:, 1], self.col_term: self.terms})
        print(self.mf)


    def dist(self, x, y):
        '''
        Parameters
        ----------
         x: array 1d
           coordenate with p1
         y: array 1d
            coordenate with p2
        Return
        ------
         euclidian distance between x and  y
        '''
        return np.sqrt(np.sum((x - y) ** 2))


    def pertinence(self, x_test, plano=1):
        '''
        Parameters
        ----------
         x_test: array 2d
           time series x(t) and x'(t)

        Return
        ------
         pertinence and term
        '''
        p_max = 0
        t_max = None
        all_p = []

        for j in range(self.mf.shape[0]):

            pert = []

            # change where
            cj = 0
            if plano == 1:
                cj = self.mf[self.col].iloc[j]
            else:
                cj = self.mf[[self.col, self.col_imf]].iloc[j].values

            num = self.dist(x_test, cj)

            for k in range(self.mf.shape[0]):

                # change where

                if plano == 1:
                    ck = self.mf[self.col].iloc[k]
                else:
                    ck = self.mf[[self.col, self.col_imf]].iloc[k].values

                dem = self.dist(x_test, ck)

                pert.append((num / dem) ** 2)

            p = 1 / np.sum(pert)
            # print(self.mf[self.col_term].iloc[j],'->', p)
            all_p.append(p)
            if p > p_max:
                p_max = p
                t_max = self.mf[self.col_term].iloc[j]

        return t_max, p_max


    def fuzzify_ts(self):
        '''
        attach a fuzzy set to each observation of the time series X'(t)
        '''
        self.ts_terms = [self.pertinence(x_test=self.ts_array[i])[0] for i in range(self.partition + 1)]


    def create_rules(self):
        '''
        :param ts_terms: time series with linguistics terms ([A1, A2, ..., AN])
        :return: base of rules
        '''

        flrg = {}
        for termo in set(self.mf[self.col_term]): flrg[termo] = []
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
                y_pred_value = self.mf[self.mf[self.col_term] == antecedente][self.col].values[0]

            else:
                pred = [self.mf[self.mf[self.col_term] == consequente][self.col].values[0] for consequente in
                        self.flrg[antecedente]]
                y_pred_value = np.mean(pred)

            y_pred.append(y_pred_value)

        return y_pred


    def MidPointInTest(self, ):
        '''
        Defuzzification with Midpoint based by [1]
        forecasting term ans value
        :return: y_pred - vector of numerbs with prediction
        '''
        y_pred = []

        antecedente = self.ts_terms[self.partition]

        for i in range(len(self.ts_array) - self.partition):

            y_pred_value = 0

            if len(self.flrg[antecedente]) == 0: 
                y_pred_value = self.mf[self.mf[self.col_term] == antecedente][self.col].values[0]

            else: 
                pred = [self.mf[self.mf[self.col_term] == consequente][self.col].values[0] for consequente in
                        self.flrg[antecedente]]
                y_pred_value = np.mean(pred)

            y_pred.append(y_pred_value)

            antecedente = self.pertinence(round(y_pred_value, 5))[0]

        return y_pred

    def fit(self, ts_x, ts_y, C=7, train=1, log=False):
        '''
        Parametes
        ---------
        ts_x: array or list (N)
           - with time series values
        ts_y: array or list (N)
           - with time series values
        u: matrix (N, C) where C is the numbers of clusters
          - pertinence matriz
        train: int (optional)
          - partition train ans test
        association: str  - 'fuzzy' or 'crip'
          - method for association data to labels
        '''

        self.ts_array = np.array(ts_x)
        self.ts_imfs = np.array(ts_y)

        self.X = np.array([ts_x, ts_y]).T
        self.C = C

        # cluster with FCM and define membership function
        self.clustering_FCM()

        # partition train and test
        if train == 1:
            self.partition = len(self.ts_array) - 1
        else:
            self.partition = int(len(self.ts_array) * train)

        # fuzzify observations
        self.fuzzify_ts()

        # create rules
        self.flrg = self.create_rules()

    def summary(self, ):
        print('Conjuntos:\n', self.clusters)
        print('Membership Funtion:\n', self.mf)
        print('Terms:\n', set(self.mf['term']))
        print('TS TERMS: \n', self.ts_terms)
        print('Midpoints:\n', self.md)
        print('-' * 30)


    def predict(self, plot=True, SM=True):

        val = validation()

        if self.partition is (len(self.ts_array) - 1):
            y_true = self.ts_array[1:]
            y_pred = self.MidPoint()

            # visualize
            mape, mae, mse, rmse, dtw = val.validation_forecasting(_y_true=y_true.copy(),
                                                                   _y_pred=y_pred.copy(),
                                                                   plot=plot,
                                                                   points=True,
                                                                   decimal=2,
                                                                   show_metrics=SM)
            return mape, mae, mse, rmse, dtw

        # forecasting in test partition
        else:
            y_true = self.ts_array[self.partition:]
            y_pred = self.MidPointInTest()

            mape, mae, mse, rmse, dtw = val.validation_forecasting(_y_true=y_true,
                                                                   _y_pred=y_pred,
                                                                   plot=True,
                                                                   decimal=2,
                                                                   show_metrics=SM)
            if plot:
                val.plot_forecasting_and_ts(self.ts_array, y_pred, self.partition,
                                            np.arange(self.partition, len(self.ts_array)))

            return mape, mae, mse, rmse, dtw


class Auto_MV_FCMB_LEE:

    def __init__(self):
            '''
            Model:
            ------
            - Time-invariant first order
            - Uses the rules and defuzzification model proposed by Chen 1996
            - Performs the prediction of the linguistic term and numerical value
            '''

    def clustering_FCM(self):

        self.fcm = FCM(n_clusters=self.C, m=2)
        self.fcm.fit(self.X)

        # outputs
        self.fcm_centers = self.fcm.centers
        self.u = self.fcm.u
        self.fcm_labels = self.fcm.u.argmax(axis=1)

        self.col = 'ts'
        self.col_imf = 'imfs'
        self.col_term = 'terms'

        # create membership function
        self.terms = ['A.' + str(i) for i in range(self.C)]

        self.mf = pd.DataFrame(
            {self.col: self.fcm_centers[:, 0], self.col_imf: self.fcm_centers[:, 1], self.col_term: self.terms})
        print(self.mf)


    def dist(self, x, y):
        '''
        Parameters
        ----------
         x: array 1d
           coordenate with p1
         y: array 1d
            coordenate with p2
        Return
        ------
         euclidian distance between x and  y
        '''
        return np.sqrt(np.sum((x - y) ** 2))


    def pertinence(self, x_test, plano=1):
        '''
        Parameters
        ----------
         x_test: array 2d
           time series x(t) and x'(t)

        Return
        ------
         pertinence and term
        '''
        p_max = 0
        t_max = None
        all_p = []

        for j in range(self.mf.shape[0]):

            pert = []

            # change where
            cj = 0
            if plano == 1:
                cj = self.mf[self.col].iloc[j]
            else:
                cj = self.mf[[self.col, self.col_imf]].iloc[j].values

            num = self.dist(x_test, cj)

            for k in range(self.mf.shape[0]):

                # change where

                if plano == 1:
                    ck = self.mf[self.col].iloc[k]
                else:
                    ck = self.mf[[self.col, self.col_imf]].iloc[k].values

                dem = self.dist(x_test, ck)

                pert.append((num / dem) ** 2)

            p = 1 / np.sum(pert)
            # print(self.mf[self.col_term].iloc[j],'->', p)
            all_p.append(p)
            if p > p_max:
                p_max = p
                t_max = self.mf[self.col_term].iloc[j]

        return t_max, p_max


    def fuzzify_ts(self):
        '''
        attach a fuzzy set to each observation of the time series X'(t)
        '''
        self.ts_terms = [self.pertinence(x_test=self.ts_array[i])[0] for i in range(self.partition + 1)]


    def create_rules(self):
        '''
        :param ts_terms: time series with linguistics terms ([A1, A2, ..., AN])
        :return: base of rules
        '''

        flrg = {}
        for termo in set(self.mf[self.col_term]): flrg[termo] = []

        for i in range(self.partition):
            antecedente = self.ts_terms[i]
            consequente = self.ts_terms[i + 1]
            if consequente not in flrg[antecedente]: flrg[antecedente].append(consequente)

        return flrg


    def defuzzification_LEE(self, consequentes):
        """
        Step 8: Assigning weights.
        consequentes: list with consequent terms
        return: defuzzyfied value
        """

        weigths = np.arange(1, (len(consequentes) + 1), dtype=np.float32)
        weigths /= np.sum(weigths)

        mid_points = []
        for term_cons in consequentes: mid_points.append(
            float(self.mf[self.mf[self.col_term] == term_cons][self.col].values[0]))
        cons_midpoints = np.array(mid_points)

        return sum(weigths * cons_midpoints.T)


    def MidPoint_Weighted(self, ):
        '''
        Defuzzification with Midpoint create by [1]
        next terms can be recognized - forecasting only value
        :param flrg: rules base (A.1:[terms], A.2:[terms], ..., A.n:[terms]) lista encadeada
        :param mf(membership function): dict with terms of sets and yours midpoints (A1:midpoint,..., An:midpoint)
        :return: y_pred - vector of numerbs with prediction
        '''
        print('CONVENCIONAL')
        y_pred = []

        for i in range(self.partition - 1):
            antecedente = self.ts_terms[i]
            value = self.defuzzification_LEE(self.flrg[antecedente])
            y_pred.append(value)
        return y_pred


    def MidPointInTest_Weighted(self, ):
        '''
        Defuzzification with Midpoint based by [1]
        forecasting term ans value20.48
        :return: y_pred - vector of numerbs with prediction
        '''

        y_pred = []

        antecedente = self.ts_terms[self.partition]

        # partition_test = len(self.ts_terms) - self.partition
        # print('Partição de test:')
        for i in range(self.ts_array.shape[0] - self.partition):

            if len(self.flrg[antecedente]) == 0:  # verify if antecedent not contais consequent
                y_pred_value = self.defuzzification_LEE([antecedente])
            else:
                y_pred_value = self.defuzzification_LEE(self.flrg[antecedente])

            y_pred.append(y_pred_value)

            # fazer a previsão do termo
            antecedente = self.pertinence(round(y_pred_value, 5))[0]

        return y_pred


    def fit(self, ts_x, ts_y, C=7, train=1, log=False):
        '''
        Parametes
        ---------
        ts_x: array or list (N)
           - with time series values
        ts_y: array or list (N)
           - with time series values
        u: matrix (N, C) where C is the numbers of clusters
          - pertinence matriz
        train: int (optional)
          - partition train ans test
        association: str  - 'fuzzy' or 'crip'
          - method for association data to labels
        '''

        self.ts_array = np.array(ts_x)
        self.ts_imfs = np.array(ts_y)

        self.X = np.array([ts_x, ts_y]).T
        self.C = C

        self.clustering_FCM()

        # partition train and test
        if train == 1:
            self.partition = len(self.ts_array) - 1
        else:
            self.partition = int(len(self.ts_array) * train)

        # fuzzify observations
        self.fuzzify_ts()

        # create rules
        self.flrg = self.create_rules()


    def summary(self, ):
        print('Conjuntos:\n', self.clusters)
        print('Membership Funtion:\n', self.mf)
        print('Terms:\n', set(self.mf['term']))
        print('TS TERMS: \n', self.ts_terms)
        print('Midpoints:\n', self.md)
        print('-' * 30)
        

    def predict(self, plot=True, SM=True):

        val = validation()

        if self.partition is self.ts_array.shape[0]:
            y_true = self.ts_array[1:]
            y_pred = self.MidPoint_Weighted()

            # visualize
            mape, mae, mse, rmse, dtw = val.validation_forecasting(_y_true=y_true.copy(),
                                                                   _y_pred=y_pred.copy(),
                                                                   plot=plot,
                                                                   points=True,
                                                                   decimal=2,
                                                                   show_metrics=SM)
            return mape, mae, mse, rmse, dtw

        # forecasting in test partition
        else:
            y_true = self.ts_array[self.partition:]
            y_pred = self.MidPointInTest_Weighted()

            self.y_pred = y_pred
            self.y_true = y_true

            mape, mae, mse, rmse, dtw = val.validation_forecasting(_y_true=y_true,
                                                                   _y_pred=y_pred,
                                                                   plot=True,
                                                                   decimal=2,
                                                                   show_metrics=SM)
            if plot:
                val.plot_forecasting_and_ts(self.ts_array, y_pred, self.partition,
                                            np.arange(self.partition, len(self.ts_array)))

            return mape, mae, mse, rmse, dtw