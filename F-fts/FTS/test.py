
import pandas as pd
import numpy as np

from Models.Chen import Chen1996

from Models.Lee import Lee2009

from Models.Wang_Chen import Wang2013_Chen
from Models.Wang_Lee import Wang2013_Lee

from Models.MV_Chen import STFMV_Convencional, STFMV_Auto

from Models.MV_Lee import STFMV_Convencional as my_fts_lee
from Models.MV_Lee import STFMV_Auto as my_fts_lee_auto

from PyEMD import EMD



def Chen_with_Alabama():
    ts = pd.read_csv('/home/nigom/mata31/data/csv/Enrollments.csv', sep=';', index_col=[0])
    model = Chen1996()
    model.fit(ts['Enrollments'].values, sets=7, d1=55, d2=663, train=0.8)
    model.summary()
    model.predict()



def Lee_with_Alabama():
    ts = pd.read_csv('/home/nigom/mata31/data/csv/Enrollments.csv', sep=';', index_col=[0])
    model = Lee2009()
    model.fit(ts['Enrollments'].values, sets=7, d1=55, d2=663)
    model.summary()
    model.predict()


def Wang_with_Alabama():
    ts = pd.read_csv('/home/nigom/mata31/data/csv/Enrollments.csv', sep=';', index_col=[0])
    model = Wang2013_Chen()
    model.fit(ts['Enrollments'].values, sets=7, d1=55, d2=663)
    model.predict()



def Wang_with_Alabama_Lee():
    ts = pd.read_csv('/home/nigom/mata31/data/csv/Enrollments.csv', sep=';', index_col=[0])
    model = Wang2013_Lee()
    model.fit(ts['Enrollments'].values, sets=7, d1=55, d2=663)
    model.predict()



def MV_model_Alabama_Chen():
    X = np.array([[13055., 13146.],
       [13563., 13476.],
       [13867., 13994.],
       [14696., 14563.],
       [15460., 15128.],
       [15311., 15652.],
       [15603., 16083.],
       [15861., 16403.],
       [16807., 16576.],
       [16919., 16508.],
       [16388., 16143.],
       [15433., 15655.],
       [15497., 15280.],
       [15145., 15256.],
       [15163., 15574.],
       [15984., 16152.],
       [16859., 16869.],
       [18150., 17594.],
       [18970., 18197.],
       [19328., 18559.],
       [19337., 18597.],
       [18876., 18317.]])


    model = STFMV_Convencional()
    model.fit(X=X, k=7)
    model.predict()


def MV_model_Alabama_Chen_Auto():


    # data
    ts = pd.read_csv('/home/marcos/mata31/data/csv/Enrollments.csv', sep=';', index_col=[0])
    col = ts.keys().to_list()[0]

    emd = EMD()
    imfs = emd.emd(ts[col].values)

    model = STFMV_Auto()
    #model.fit(ts_x=ts[col].values, ts_y=imfs[1]+imfs[2], C=5,  train=1)
    model.fit(ts_x=ts[col].values, ts_y=imfs[1] + imfs[2], C=20, association='fuzzy', train=1)
    model.predict()

def MV_model_Alabama_Lee():
    X = np.array([[13055., 13146.],
       [13563., 13476.],
       [13867., 13994.],
       [14696., 14563.],
       [15460., 15128.],
       [15311., 15652.],
       [15603., 16083.],
       [15861., 16403.],
       [16807., 16576.],
       [16919., 16508.],
       [16388., 16143.],
       [15433., 15655.],
       [15497., 15280.],
       [15145., 15256.],
       [15163., 15574.],
       [15984., 16152.],
       [16859., 16869.],
       [18150., 17594.],
       [18970., 18197.],
       [19328., 18559.],
       [19337., 18597.],
       [18876., 18317.]])


    model = my_fts_lee()
    model.fit(X=X, k=7)
    model.predict()


def MV_model_Alabama_Lee_Auto():


    # data
    ts = pd.read_csv('/home/nigom/mata31/data/csv/Enrollments.csv', sep=';', index_col=[0])
    col = ts.keys().to_list()[0]

    emd = EMD()
    imfs = emd.emd(ts[col].values)

    model = my_fts_lee_auto()
    #model.fit(ts_x=ts[col].values, ts_y=imfs[1]+imfs[2], C=5,  train=1)
    model.fit(ts_x=ts[col].values, ts_y=imfs[1] + imfs[2], C=7, association='fuzzy')
    model.predict()

if __name__ == '__main__':
    MV_model_Alabama_Lee_Auto()
    #MV_model_Alabama_Lee()
    #MV_model_Alabama_Chen_Auto()
    #MV_model_Alabama_Chen()