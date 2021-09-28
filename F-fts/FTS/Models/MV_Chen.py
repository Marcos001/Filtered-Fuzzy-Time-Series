
from fcmeans import FCM
from FTS.Evaluate.Validation import validation
import numpy as np
import pandas as pd

class STFMV_Convencional_Chen:

    def __init__(self):
        '''
        Model:
        ------
         - Time-invariant first order
         - Uses the rules and defuzzification model proposed by Chen 1996
         - It performs the prediction of the numerical value only, since it knows the linguistic term
        '''

    def __clustering_FCM(self):

        self.fcm = FCM(n_clusters=self.K, m=2)
        self.fcm.fit(self.X)

        # X[:,0] ts
        # X[:,1] imf

        # outputs
        self.fcm_centers = self.fcm.centers
        self.u = self.fcm.u
        self.fcm_labels = self.fcm.u.argmax(axis=1)


    def predict_term_fcm(self, test):
        indice = self.fcm.predict(test)[0]
        return 'A.' + str(indice + 1)


    def __create_fuzzy_sets(self,):

        self.__clustering_FCM()

        self.df_fuzzy_sets = {}
        for i in range(self.fcm_centers.shape[0]):
            self.df_fuzzy_sets[self.predict_term_fcm(self.fcm_centers[i])] = self.fcm_centers[i,0]



    def gaussiana(self, value, m, sigma):
        """
        define pertinence y of the trapezoidal function from x values
        x: value crisp
        m: middle value
        sigma: statard desviation
        return: value of pertinence [0,1]
        """
        return np.exp(-((value - m) ** 2) / (sigma ** 2))


    def fuzzify_ts(self):
        self.ts_terms = [self.predict_term_fcm(self.X[i]) for i in range(self.X.shape[0]-1)]


    def create_rules(self):
        '''
        Creating dictionary to store relationship groups
        '''
        self.fuzzify_ts()

        self.flrg = {}
        for termo in list(self.df_fuzzy_sets.keys()): self.flrg[termo] = []

        for i in range(self.X.shape[0]-2):
            antecedente = self.ts_terms[i]
            consequente = self.ts_terms[i + 1]
            if consequente not in self.flrg[antecedente]: self.flrg[antecedente].append(consequente)


    def fit(self, X, k, log=False):
        '''
        Parametes
        ---------
        X: array data 2D (intances, 2)
          -  ([[time series][IMFs]])
        K: int
          - Number of fuzzy sets
        '''
        self.X = X
        self.K = k
        self.__create_fuzzy_sets()
        self.create_rules()

        if log:
            self.summary()


    def summary(self):
        print()
        print('Fuzzy Sets: \n',self.df_fuzzy_sets)
        print('RULES:\n', self.flrg)


    def predict(self, plot=True, SM=True):
        y_true = self.X[1:, 0]
        y_pred = self.MidPoint()

        # visualize
        mape, mae, mse, rmse, dtw = validation().validation_forecasting(_y_true=y_true.copy(),
                                                                   _y_pred=y_pred.copy(),
                                                                        plot=plot,
                                                                        points=True,
                                                                        decimal=2,
                                                                        show_metrics=SM)
        return mape, mae, mse, rmse, dtw


    def MidPoint(self, ):
        '''
        Defuzzification with Midpoint create by [1]
        next terms can be recognized - forecasting only value
        :param flrg: rules base (A.1:[terms], A.2:[terms], ..., A.n:[terms]) lista encadeada
        :param mf(membership function): dict with terms of sets and yours midpoints (A1:midpoint,..., An:midpoint)
        :return: y_pred - vector of numerbs with prediction
        '''

        y_pred = []

        for i in range(self.X.shape[0]-1):

            antecedente = self.ts_terms[i]
            y_pred_value = 0

            if len(self.flrg[antecedente]) == 0:
                y_pred_value = self.df_fuzzy_sets[antecedente]

            else: # replace
                pred = [self.df_fuzzy_sets[consequente] for consequente in self.flrg[antecedente]]
                y_pred_value = np.mean(pred)

            y_pred.append(y_pred_value)

        return y_pred


class STFMV_Auto_Chen:

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

        # X[:,0] ts
        # X[:,1] imf

        # outputs
        self.fcm_centers = self.fcm.centers
        self.u = self.fcm.u
        self.fcm_labels = self.fcm.u.argmax(axis=1)


    def associate_data_to_label_fuzzy(self, ts_array, u):
        """
        Create crypt sets sorted by grouped data
        Parameters
        ----------
            ts_array: ([N]) array 1d for time series
            u: matriz (N, C), where C is numbers of clusters
        Return
        ------
           groups containing the data values associated with the grouping labels
        """
        labels = np.arange(u.shape[1])
        limiar = 1 / u.shape[1]
        grupos = {}
        for label in labels: grupos[label] = []

        for i in range(u.shape[0]):  # N
            for j in range(u.shape[1]):  # C
                if u[i][j] >= limiar:
                    grupos[j].append(ts_array[i])
        return grupos


    def associate_data_to_label_crip(self, ts_array, y_pred):
        """
        Create crypt sets sorted by grouped data
        Parameters
        ----------
            ts_array: array or list ([N])  for time series
            y_pred: array or list (N), contains labels to data in ts_array
        Return
        ------
           groups containing the data values associated with the grouping labels
        """

        labels = list(set(y_pred))
        data = {}
        for label in labels: data[label] = []
        for i in range(len(ts_array)):
            data[y_pred[i]].append(ts_array[i])
        return data


    def build_membership_function(self, sets, log=False):
        """
        modeling the Gaussian function for crip sets
        sets: Dictionary with each set {label0:[set0], label1:[set1],...,labelN:[setN]}
        return: dataframe with the Gaussian parameters of each set
        """
        parametros = []

        for i in sets:

            dici = {'term': 'A.' + str(i),
                    'm': np.mean(sets[i]),
                    'std': np.std(sets[i]),
                    'min': np.min(sets[i]),
                    'max': np.max(sets[i])
                    }

            if log:
                print(dici)
            parametros.append(dici)

        return pd.DataFrame(parametros)


    def __create_fuzzy_sets(self):

        self.clusters = {}
        conjuntos_y = {}

        if self.method is 'fuzzy':
            self.clusters = self.associate_data_to_label_fuzzy(self.ts_array, self.u)
            conjuntos_y = self.associate_data_to_label_fuzzy(self.ts_imfs, self.u)
        else:
            self.clusters = self.associate_data_to_label_crip(self.ts_array, self.u.argmax(axis=1))
            conjuntos_y = self.associate_data_to_label_crip(self.ts_imfs, self.u.argmax(axis=1))

        # merge x and y
        [self.clusters[i].append(conjuntos_y[i][j]) for i in self.clusters for j in range(len(conjuntos_y[i]))]

        # create the membership function parameters
        self.mf = self.build_membership_function(self.clusters)

        # create dict with midpoints
        self.md = {}
        for i in range(self.mf.shape[0]):
            self.md[self.mf['term'].iloc[i]] = self.mf['m'].iloc[i]


    def gaussiana(self, value, m, sigma):
        """
        define pertinence y of the trapezoidal function from x values
        x: value crisp
        m: middle value
        sigma: statard desviation
        return: value of pertinence [0,1]
        """
        return np.exp(-((value - m) ** 2) / (sigma ** 2))


    def fuzzify(self, value, term=False, verbose=False):
        """
        value: crip value to be calculated for membership
        term (False): All membership values belonging to the N groups {type df pandas}
        term (True): Returns the linguistic term of the highest membership value {
        pertinence (float), termo (string)
        }
        """
        perts = []

        for i in range(self.C):

            dici_resposta = {'p': self.gaussiana(value, m=self.mf['m'][i], sigma=self.mf['std'][i]),
                             't': self.mf['term'][i]}
            perts.append(dici_resposta)

            if verbose:
                print(dici_resposta)

        df = pd.DataFrame(perts)

        # return max pert
        if term: # return pertinence and linguistic term
            indice_find = int(df[df['p'] == max(df['p'])].index.values)
            return df['p'].iloc[indice_find], df['t'].iloc[indice_find]

        return df


    def fuzzify_ts(self):
        self.ts_terms = [self.fuzzify(value=self.ts_array[i], term=True)[1] for i in range(self.partition+1)]


    def create_rules(self):
        '''
        :param ts_terms: time series with linguistics terms ([A1, A2, ..., AN])
        :return: base of rules
        '''
        # creating dictionary to store relationship groups
        flrg = {}
        for termo in set(self.mf['term']): flrg[termo] = []

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

        for i in range(len(self.ts_array) - self.partition):

            y_pred_value = 0

            if len(self.flrg[antecedente]) == 0:
                y_pred_value = self.md[antecedente]

            else:
                pred = [self.md[consequente] for consequente in self.flrg[antecedente]]
                y_pred_value = np.mean(pred)

            y_pred.append(y_pred_value)

            antecedente = self.fuzzify(y_pred_value, term=True)[1]

        return y_pred


    def fit(self, ts_x, ts_y, C=7, train=1, association='fuzzy',  log=False):
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
        self.ts_imfs  = np.array(ts_y)

        self.X = np.array([ts_x, ts_y]).T
        self.C = C
        self.method = association

        self.clustering_FCM()

        self.__create_fuzzy_sets()

        if train == 1:
            self.partition = len(self.ts_array) - 1
        else:
            self.partition = int(len(self.ts_array) * train)

        self.fuzzify_ts()

        self.flrg = self.create_rules()


    def summary(self,):
        print('Sets:\n', self.clusters)
        print('Membership Funtion:\n',self.mf)
        print('Terms:\n', set(self.mf['term']))
        print('TS TERMS: \n', self.ts_terms)
        print('Midpoints:\n', self.md)
        print('-'*30)


    def predict(self, plot=True, SM=True):

        val = validation()

        if self.partition is (len(self.ts_array)-1):
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

        else:
            y_true = self.ts_array[self.partition:]
            y_pred = self.MidPointInTest()



            mape, mae, mse, rmse, dtw = val.validation_forecasting(_y_true=y_true,
                                                              _y_pred=y_pred,
                                                              plot=True,
                                                              decimal=2,
                                                              show_metrics=SM)
            if plot:
                val.plot_forecasting_and_ts(self.ts_array, y_pred, self.partition, np.arange(self.partition, len(self.ts_array)))

            return mape, mae, mse, rmse, dtw
