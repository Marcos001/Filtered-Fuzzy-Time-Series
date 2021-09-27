
'''
compute the fcm memberships in 1d and 2d
'''

import numpy as np

class FCMB:

    def __init__(self, df_centers, log=False):
        '''

        Parameters
        ----------
        fcm_centers: 2d centers cluster
            dataframe d2 pandas
        '''

        ''' columms '''
        self.col, self.col_imf = list(df_centers.keys())
        self.col_term = 'terms'

        self.mf = df_centers.copy()

        # create linguistic terms
        self.terms = ['A.' + str(i) for i in range(self.mf.shape[0])]

        self.mf[self.col_term] = self.terms

        if log:
            self.summary()


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
            if plano == 1: cj = self.mf[self.col].iloc[j]
            else: cj = self.mf[[self.col, self.col_imf]].iloc[j].values

            num = self.dist(x_test, cj)

            for k in range(self.mf.shape[0]):

                # change where

                if plano == 1: ck = self.mf[self.col].iloc[k]
                else: ck = self.mf[[self.col, self.col_imf]].iloc[k].values

                dem = self.dist(x_test, ck)

                pert.append((num / dem) ** 2)

            p = 1 / np.sum(pert)

            all_p.append(p)
            if p > p_max:
                p_max = p
                t_max = self.mf[self.col_term].iloc[j]

        return t_max, p_max



    def predict(self, x, term=True):
        '''
        calculate pertinence function and associate to cluster
        '''

        x = np.array(x)
        p = 0
        t = None
        if x.shape[0] == 1:
            t, p = self.pertinence(x)
        else:
            t, p = self.pertinence(x, plano=2)

        if term:
            return t, p
        else:
            return p

    def summary(self, ):
        print('=' * 30)
        print('SUMMARY')
        print('=' * 30)
        print('col_1:{} and col_2:{}'.format(self.col, self.col_imf))
        print('-' * 30)
        print(self.mf)