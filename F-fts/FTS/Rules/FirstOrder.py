
'''
class que create fuzzy rules with first order
'''

class FirstOrder:

    def __init__(self):
        pass

    def create_rules(self, ts_terms):
        '''
        :param ts_terms: time series with linguistics terms ([A1, A2, ..., AN])
        :return: base of rules
        '''
        flrg = {}
        for termo in set(ts_terms): flrg[termo] = []

        for i in range(ts_terms.shape[0] - 1):
            ante = ts_terms[i]
            cons = ts_terms[i + 1]
            if cons not in flrg[ante]: flrg[ante].append(cons)

        return flrg