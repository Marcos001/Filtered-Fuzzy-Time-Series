import pandas as pd
import numpy as np


class FirtOrder:

    def __init__(self, ):
        '''
        first order rules based on Chen 1996
        '''

    def train(self, df_fts, p_train):
        self.terms = sorted(set(df_fts['term']))
        self.size = len(self.terms)  # quantidade de termos, conjuntos
        self.df_flrg = pd.DataFrame(np.zeros([self.size, self.size], dtype=np.uint8), index=self.terms,
                                    columns=self.terms)
        self.create_rules(df_fts, p_train)


    def create_rules(self, df_fts, p_train):
        # train
        for i in range(p_train):
            ANT = df_fts['term'].iloc[i]
            CON = df_fts['term'].iloc[i + 1]
            self.df_flrg[CON][ANT] = 1
        self.df_flrg


    def inference(self, df_fts, col):

        y_true = []
        y_pred = []

        for i in range(self.p_train - 1, self.df_fts.shape[0] - 1):

            v_obs = df_fts[col].iloc[i]
            t_ant = df_fts['term'].iloc[i]

            if len(self.df_flrg[self.df_flrg[t_ant] == 0]) == self.size:
                print('Sem consequentes!')
                y_pred.append(float(df_gauss['m'][df_gauss['term'] == t_ant]))
                y_true.append(df_fts[col].iloc[i + 1])
            else:
                pred = []
                for con in termos:
                    if df_RLF[con][t_ant] == 1:
                        pred.append(float(df_gauss['m'][df_gauss['term'] == con]))
                y_pred.append(np.mean(pred))
                y_true.append(df_fts[col].iloc[i + 1])

        validation_forecasting(y_true.copy(), y_pred.copy(), points=True)


class Weighted:

    def __init__(self, ):
        pass


class FuzzyRules:

    def __init__(self, type=1):
        '''
        type: 1 (FirtOrder), 2 (Weighted)
        '''
        self.rules_type = type
        self.model_rules = FirtOrder()

        pass

    def fit(self, df_fts, col=None, p_train=0.8):
        '''
        df_fts: dataframe pandas with  columns [time, obs, terms]
        '''

        if col == None: col = df_fts.columns.to_list()[0]

        self.p_train = int(df_fts.shape[0] * p_train)
        self.model_rules.train(df_fts, self.p_train)

        print('Created rules:')
        print(self.model_rules.df_flrg)

    def inference(self, ):
        pass
