import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.style.use('ggplot')

class validation:

    def __init__(self):
        '''
        validation forecasting with y_true and y_pred
        '''


    def mean_absolute_percentage_error(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


    def validation_forecasting(self, _y_true, _y_pred, plot=True, points=False, decimal=3, show_metrics=True):
        '''
        calculate regression metrics to measure prediction.
        '''
        mape = round(self.mean_absolute_percentage_error(np.array(_y_true), np.array(_y_pred)), decimal)
        mae = round(mean_absolute_error(_y_true, _y_pred), decimal)
        mse = round(mean_squared_error(_y_true, _y_pred), decimal)
        rmse = round(np.math.sqrt(mean_squared_error(_y_true, _y_pred)), decimal)
        d, cost_matrix, acc_cost_matrix, path = dtw(_y_true, _y_pred, dist=lambda x, y: np.abs(x - y))

        if show_metrics:
            print('=' * 30)
            print('MAPE......: {}'.format(mape))
            print('-' * 30)
            print('MAE.......: {}'.format(mae))
            print('-' * 30)
            print('MSE.......: {}'.format(mse))
            print('-' * 30)
            print('RMSE......: {}'.format(rmse))
            print('-' * 30)
            print('DTW.......: {}'.format(d))
            print('=' * 30)

        if plot:
            self.plot_all(_y_true, _y_pred, acc_cost_matrix, path, d)
            #self.plot_forecasting(_y_true, _y_pred, points)
            #self.plot_dtw(acc_cost_matrix, path, d)
        return mape, mae, mse, rmse, d


    def plot_forecasting(self, _y_true, _y_pred, p):
        plt.figure(figsize=(12, 6))
        if p:
            plt.plot(_y_true, '-o', label='TS')
            plt.plot(_y_pred, '-o', label='Predict')
        else:
            plt.plot(_y_true, label='TS')
            plt.plot(_y_pred, label='Predict')
        
        plt.ylabel('Values')
        plt.xlabel('Time \n\n (A)')
        plt.title('FTS Prediction')
        plt.legend()
        plt.show()


    def plot_forecasting_and_ts(self, ts, y_pred, _p_train, indices):
        plt.figure(figsize=(12, 8))
        plt.title('Forecasting - Partition of the train and test')
        plt.plot(ts, '-o')
        plt.plot(indices, y_pred, '-o')
        plt.ylabel('Values')
        plt.xlabel('Time \n\n (A)')
        plt.axvline(_p_train, c='k')
        plt.show()


    def plot_dtw(self, acc_cost_matrix, path, d):
        plt.title('DTW: {}'.format(d))
        plt.grid(False)
        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.ylabel('Time')
        plt.xlabel('Time \n\n (B)')
        plt.show()
        

    def plot_all(self, _y_true, _y_pred, acc_cost_matrix, path, d, save=True):
        
        fig, ax = plt.subplots(1, 2, figsize=(15,5), gridspec_kw={'width_ratios': [2.5, 1]})
        
        ax[0].plot(_y_true, label='Expected')
        ax[0].plot(_y_pred, label='Predicted')
        ax[0].set_ylabel('Enrollments')
        ax[0].set_xlabel('Time \n\n (A)')
        ax[0].legend(fontsize=14)
        
        ax[1].grid(False)
        ax[1].imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        ax[1].plot(path[0], path[1], 'w')
        ax[1].set_ylabel("Predicted")
        ax[1].set_xlabel("Expected \n\n (B)")
        ax[1].legend()
        
        if save:
            plt.savefig('/home/ds/app/out/forecasting.pdf', bbox_inches='tight')
        
        plt.show()


