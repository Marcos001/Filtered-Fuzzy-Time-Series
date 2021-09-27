''' libs '''

import os, warnings
import numpy as np
import pandas as pd
from PyEMD import EMD
from fcmeans import FCM
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def norm(x):
    ''' Min-Max Feature scaling '''
    return (x - min(x)) / (max(x) - min(x))

def standardization(x):
    return x - np.mean(x) /  np.std(x)

def isodd(number):
    if (number % 2) == 0: return False
    return True


def create_sets_crips(data_ts, y_pred):
    """
    Create crypt sets sorted by grouped data
    """
    labels = list(set(y_pred))
    data = {}

    for label in labels: data[label] = []

    for i in range(len(data_ts)):
        data[y_pred[i]].append(data_ts[i])

    return data


def gaussiana(x, m, sigma, max_degree=None):
    """
    define pertinence y of the trapezoidal function from x values
    x: value to get pertinence
    m: mean value
    sigma: sigma
    
    retorna o valor da pertinência
    """
    return np.exp(-((x-m)**2)/(sigma**2))


def get_pertinences(x, _para, desc=False, verbose=False):
    """
    x: valor crip a ser calculado a pertinência
    _para: parameters of each membership function
    
    desc (False): All membership values belonging to the N groups {type df pandas}
    desc (True): Returns the linguistic term of the highest membership value {
     membership (float), term (string)
    }
    """
    
    perts = []
    
    for i in range(C):
        
        dici_resposta = {'p':gaussiana(x, m=_para['m'][i],sigma=_para['std'][i]),
                         't':_para['term'][i]}
        
        perts.append(dici_resposta)
        
        if verbose:
            print(dici_resposta)
    
    df = pd.DataFrame(perts)
    
    # return max pert
    if desc:
        indice_find = int(df[df['p'] == max(df['p'])].index.values)
        return   df['p'].iloc[indice_find], df['t'].iloc[indice_find]
    
    return df


def visualize_fuzification(crips_sets, col, df_fts):

    fig = plt.figure(figsize=(15,8))

    min_x = np.min(df_fts.index)
    max_x = np.max(df_fts.index)
    
    for i in range(df_fts.shape[0]):
        plt.annotate(df_fts['terms'].iloc[i], (df_fts.index[i], df_fts[col].iloc[i]))

    plt.plot(df_fts[col], '-o')
    for i in range(len(crips_sets)):
        plt.fill([min_x,min_x,max_x,max_x], [min(crips_sets[i]),
                                             max(crips_sets[i]), 
                                             max(crips_sets[i]), 
                                             min(crips_sets[i])], c=cores[i], alpha=0.5)
    
    
    
    plt.show()


def create_params_gaussiana(sets, log=False):
    parametros = []

    for i in range(len(sets)):
        
        dici = {'term': 'A.'+str(i+1),
                'm':np.mean(sets[i]),
                'std':np.std(sets[i]),
                'min':min(sets[i]), 
                'max':max(sets[i])
               }
        if log:
            print(dici)
        parametros.append(dici)
    
    return pd.DataFrame(parametros)



def rebuild_fuzzy_sets(sets, log=False):
    """
    modeling the Gaussian function for crip sets
    sets: Dictionary with each set {label0:[set0], label1:[set1],...,labelN:[setN]}
    return: dataframe with the Gaussian parameters of each set
    """
    parametros = []
    
    for i in sets:
        
        dici = {'term': 'A.'+str(i),
                'm':np.mean(sets[i]),
                'std':np.std(sets[i]),
                'min':min(sets[i]), 
                'max':max(sets[i])
               }

        if log:
            print(dici)
        parametros.append(dici)
    
    return pd.DataFrame(parametros)



def silhueta_fcm(data, m=2):
    ''' score silhueta 
    '''
    df_scores = {}
    for i in range(3,10):
        fcm = FCM(n_clusters=i, m=m)
        fcm.fit(data)
        # outputs
        fcm_centers = fcm.centers
        fcm_labels  = fcm.u.argmax(axis=1)

        df_scores[i] = metrics.silhouette_score(data, fcm_labels, metric='euclidean')
    return df_scores


def find_k(df_scores, ini = 3):
    '''
    plots the best number of groups with the silhouette score
    df_scores: dataframe with silhouette scores
    '''
    _max_score = max(list(df_scores.values()))
    
    for i in range(ini, len(df_scores)+ini):
        if _max_score == df_scores[i]: return i, _max_score
        

''' ======================== PLOTAGEM =============================='''

def plot_forecasting_and_ts(_df_fts, _col, _y_pred, _p_train, _indices):
    plt.figure(figsize=(12,8))
    plt.plot(_df_fts[_col])
    plt.axvline(_df_fts.index[_p_train], c='k')
    plt.plot(_indices, _y_pred)
    plt.show()


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
    

def plot_ts(ts, title=None, label=None):
    plt.figure(figsize=(12,5))
    if label:
        plt.plot(ts, label=label)
    else:
        plt.plot(ts)
    if title:
        plt.title(title)
    plt.show()


def visualize_gaussianas_cluster(df_fts, col, col_imf, u, fcm_centers, fcm_labels, SS):
    fig = plt.figure(figsize=(18,8))
    #cores = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    grupos = len(set(fcm_labels))
    
    cores = sns.color_palette(n_colors=grupos)
    
    ax = fig.add_subplot(121)
    for i in range(df_fts.shape[0]):
        ax.scatter(df_fts[col].iloc[i], df_fts[col_imf].iloc[i], color=cores[int(fcm_labels[i])])

    ax.scatter(fcm_centers[col], fcm_centers[col_imf], marker="*", s=200, c='b')
    ax.set_title('Clustering FCM com Silhueta: %.2f' %(SS))

    ax.set_xlabel('obs TS')
    ax.set_ylabel('imf')

    '''
    visualization of Gaussian membership functions on observations
    '''

    ax = fig.add_subplot(122)
    for i in range(df_fts.shape[0]):
        for j in range(grupos):
            ax.plot(df_fts[col].iloc[i], u[i][j], '*', color=cores[j])
            ax.plot(df_fts[col_imf].iloc[i], u[i][j], '*', color=cores[j])

    for i in range(grupos):
        ax.axvline(fcm_centers[col][i], c=cores[i])
        ax.annotate('A{}'.format(i+1), (fcm_centers[col][i], 0.5))
    ax.set_xlabel('Observations')
    
    ax.set_xlabel('Pertinences')
    ax.axhline((1/len(fcm_centers)), c='k')
    
    plt.show()


def plot_imfs(ts, array_imfs, n_row=2, fs=(20,20)):
    '''
    view all imfs by decomposition EMD ts
    '''
    n_col = int(array_imfs.shape[0] / 2)
    
    if isodd(array_imfs.shape[0]):  n_col += 1
    
    fig = plt.figure(figsize=fs)
    print('shape imf: ',array_imfs.shape)
    
    ax = fig.add_subplot(n_row,n_col, 1)
    ax.plot(ts)
    ax.set_title('Time Serie')
    
    for i in range(1, array_imfs.shape[0]+1):
        ax = fig.add_subplot(n_row,n_col, i+1)
        ax.plot(array_imfs[i-1])
        ax.set_title('IMF {}'.format(i-1))
        if i == (array_imfs.shape[0]): ax.set_title('Resíduo')
            
    plt.show()

    
''' ======================== VALIDAÇÂO =============================='''


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


def validation_forecasting(_y_true, _y_pred, plot=True, points=False, decimal=3):
    '''
    Calculate regression metrics to measure prediction.
    '''
    
    mape = round(mean_absolute_percentage_error(np.array(_y_true), np.array(_y_pred)), decimal)
    mae  = round(mean_absolute_error(_y_true, _y_pred), decimal)
    mse  = round(mean_squared_error(_y_true, _y_pred), decimal)
    rmse = round(np.math.sqrt(mean_squared_error(_y_true, _y_pred)), decimal) 

    print('='*30)
    print('MAPE......: {}'.format(mape))
    print('-'*30)
    print('MAE.......: {}'.format(mae))
    print('='*30)
    print('MSE.......: {}'.format(mse))
    print('-'*30)
    print('RMSE......: {}'.format(rmse))
    print('='*30)
    if plot:
        plot_forecasting(_y_true, _y_pred, points)
    return mape, mae, mse, rmse


def best_GS(df, col_mse):
    return df[df[col_mse] == min(df[col_mse])]


def verify_path(path_dir):
    if not os.path.isdir(path_dir):
        try:
            os.mkdir(path_dir)
            print('created')
        except:
            print("don't create path out")


def update_values_exp(dici, key_words, fp, log=False):

    verify_path(fp.split('/')[0])
    
    file_path = os.getcwd()+'/'+fp
    
    df = None
    
    if os.path.isfile(file_path):
        if log:
            print('read')
        df = pd.read_csv(file_path)
        df.head()
        
        oco = 0
        for i, key in enumerate(key_words):
            if dici[key] in df[key].to_list():
                oco += 1

        if oco < len(key_words):
            df = df.append(dici, ignore_index=True)
            df.to_csv(file_path, index=False)
            if log:
                print('Add new instance')
    else:
        if log:
            print('create')
        df = pd.DataFrame(dici, index=[0])
        df.to_csv(file_path, index=False)
        df.head()
