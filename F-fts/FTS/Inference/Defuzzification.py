
'''
Defuzzification Methods
 [1] - Forecasting enrollments based on fuzzy time series
'''

import numpy as np

class Defuzz:

    def __init__(self):
        '''
         method
        '''

    def MidPoint(self, ts_terms, flrg, md):
        '''
        Defuzzification with Midpoint create by [1]
        next terms can be recognized
        :param flrg: rules base (A.1:[terms], A.2:[terms], ..., A.n:[terms]) lista encadeada
        :param mf(membership function): dict with terms of sets and yours midpoints (A1:midpoint,..., An:midpoint)
        :return: y_pred - vector of numerbs with prediction
        '''

        y_pred = []

        for i in range(ts_terms.shape[0] - 1):

            # descrição linguística do termo f(t)
            antecedente = ts_terms[i]

            # valor de previsão do instante
            y_pred_value = 0

            if len(flrg[antecedente]) == 0:
                y_pred_value = md[antecedente]

            else:
                pred = [md[consequente] for consequente in flrg[antecedente]]
                y_pred_value = np.mean(pred)

            y_pred.append(y_pred_value)

        return y_pred


    def MidPointAuto(self, actual_value, flrg, md, predictor, forecasting_size=10):

        y_true = []
        y_pred = []

        antecedente = predictor(actual_value, term=True)[1]

        for i in range(forecasting_size):
            y_pred_value = 0
            if len(flrg[antecedente]) == 0:
                y_pred_value = md[antecedente]
            else:
                pred = [md[consequente] for consequente in flrg[antecedente]]
                y_pred_value = np.mean(pred)

            y_pred.append(y_pred_value)

            antecedente = predictor(y_pred_value, term=True)[1]

        return y_pred
