import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FCM_Rebuild:

    def __init__(self, X, Y):

        self.C = len(X.keys())

        self.conjuntos = X

        [self.conjuntos[i].append(Y[i][j]) for i in X for j in range(len(Y[i]))]

        self.mf, self.MF = self.rebuild_fuzzy_sets(self.conjuntos)

        self.create_samples()


    def summary(self, ):
        print('Number of sets: ', self.C)


    def get_midpoints(self):
        '''
         create dict with midpoints od the fuzzy sets
        :return: dict with terms of sets and yours midpoints (A1:midpoint,..., An:midpoint)
        '''
        midpoints = {}
        for i in range(self.mf.shape[0]): midpoints[self.mf['term'].iloc[i]] = self.mf['m'].iloc[i]
        return midpoints


    def gaussiana(self, value, m, sigma):
        """
        define pertinence y of the trapezoidal function from x values
        x: value crisp
        m: middle value
        sigma: statard desviation
        return: value of pertinence [0,1]
        """
        return np.exp(-((value - m) ** 2) / (sigma ** 2))

    
    def fuzzificar(self, value, term=False, verbose=False):
        """
        value: crip value to be calculated for membership
        term (False): All membership values belonging to the N groups {type df pandas}
        term (True): Returns the linguistic term of the highest membership value {
         membership (float), term (string)
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

        # return daframe with all pertinences
        return df



    def rebuild_fuzzy_sets(self, sets, log=False):
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

        return pd.DataFrame(parametros), pd.DataFrame(parametros, index=['A.0', 'A.1', 'A.2', 'A.3', 'A.4', 'A.5', 'A.6'])


    def create_samples(self, granulary=100):

        # create dict with name fuzzy sets
        self.data_samples = {}
        for i in range(self.mf.shape[0]):
            self.data_samples[self.mf['term'].iloc[i]] = []

        # create samples
        for i in range(self.mf.shape[0]):

            # create x and y
            U = self.mf['max'].iloc[i] - self.mf['min'].iloc[i]
            size = U / granulary

            X = []
            Y = []
            X.append(self.mf['min'].iloc[i])
            Y.append(self.gaussiana(self.mf['min'].iloc[i], m=self.mf['m'][i], sigma=self.mf['std'][i]))
            for xi in range(granulary):
                value = X[xi] + size
                X.append(value)
                Y.append(self.gaussiana(value, m=self.mf['m'][i], sigma=self.mf['std'][i]))
            self.data_samples[self.mf['term'].iloc[i]].append(X)
            self.data_samples[self.mf['term'].iloc[i]].append(Y)


    def create_samples_X(self, granulary=100):

        # create dict with name fuzzy sets
        self.x_samples = []

        # create x and y
        U = 19337 - 13055
        size = U / granulary

        self.x_samples.append(13055)

        # create samples
        for i in range(granulary):

            valor = self.x_samples[i] + size
            self.x_samples.append(valor)

        # end x samples, now, create x for C clusters

        self.y_samples_all = {}

        for i in range(self.mf.shape[0]):
            self.y_samples_all[self.mf['term'].iloc[i]] = []

        for i in range(self.mf.shape[0]):
            for j in range(len(self.x_samples)):
                p = self.gaussiana(value=self.x_samples[j], m=self.mf['m'][i], sigma=self.mf['std'][i])
                self.y_samples_all[self.mf['term'].iloc[i]].append(p)

        return self.x_samples, self.y_samples_all


    def plot_fuzzy_sets_X(self):
        plt.figure(figsize=(12,8))

        for key in self.y_samples_all:
            plt.plot(self.x_samples, self.y_samples_all[key])
        plt.show()


    def plot_fuzzy_sets(self):
        for key in self.data_samples:

            plt.plot(self.data_samples[key][0], self.data_samples[key][1])
        plt.show()



    def summary(self, ):
        print('MF:\n', self.mf)
        print('SAMPLES:\n', self.data_samples)


    # --- functions for 3d plotting

    def create_samples_X_and_Y(self, granulary=150):

        # create dict with name fuzzy sets
        self.x = []
        self.y = []

        # create x and y
        U_x = 19337 - 13055
        U_y = 18597 - 13146
        size_x = U_x / granulary
        size_y = U_y / granulary

        self.x.append(13055)
        self.y.append(13146.0)

        # create samples
        for i in range(granulary):
            valor = self.x[i] + size_x
            self.x.append(valor)

        for i in range(granulary):
            valor = self.y[i] + size_y
            self.y.append(valor)

        return self.x, self.y


    def _fuzzificar(self, value, term):
        """
        value: crip value to be calculated for membership
        term (False): All membership values belonging to the N groups {type df pandas}
        term (True): Returns the linguistic term of the highest membership value {
         membership (float), term (string)
        }
        """

        perts = []

        dici_resposta = {'p': self.gaussiana(value, m=self.MF['m'][term], sigma=self.MF['std'][term]),
                             't': self.MF['term'][term]}

        # return pertinence
        return dici_resposta['p']
