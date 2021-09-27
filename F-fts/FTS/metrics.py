import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
plt.style.use('ggplot')

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def averange_error(y_true, y_pred):
    '''
    Percentage error like Chen 1996 use.
    '''
    all_errors = []
    for i in range(len(y_true)):
        error = abs(y_true[i] - y_pred[i]) # abs after
        p_error = (error * 100) / y_true[i]
        all_errors.append(p_error)
    return np.mean(all_errors)


def plot_forecasting(_y_true, _y_pred, p):
    plt.figure(figsize=(12,6))
    if p:
        plt.plot(_y_true, '-o', label='TS')
        plt.plot(_y_pred, '-o', label='Predict')
    else:
        plt.plot(_y_true, label='TS')
        plt.plot(_y_pred, label='Predict')
    plt.legend()
    plt.show()
    

def validation_forecasting(_y_true, _y_pred, plot=True, points=False):
    '''
    - Calculate regression metrics to measure prediction.
    '''
    print('='*30)
    print('ERROR.....: %.2f%s' %(averange_error(_y_true, _y_pred), '%'))
    print('='*30)
    print('MAE.......: %.2f' %mean_absolute_error(_y_true, _y_pred))
    print('-'*30)
    print('MAPE......: {0:.2f}'.format(mean_absolute_percentage_error(np.array(_y_true), np.array(_y_pred))))
    print('='*30)
    print('MSE.......: %.2f' %mean_squared_error(_y_true, _y_pred))
    print('-'*30)
    print('RMSE......: %.2f' %np.math.sqrt(mean_squared_error(_y_true, _y_pred)))
    print('='*30)
    if plot:
        plot_forecasting(_y_true, _y_pred, points)