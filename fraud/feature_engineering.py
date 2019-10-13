import pandas as pd
import datetime
import numpy as np
from sklearn import preprocessing
import gc

#############################################
# Feature Engineering
# TODO PCA for V features
# TODO SVD for V features
#############################################

def drop_mostly_null_and_skewed_cols(train, test):
    cols_drop_train = [
        cols for cols in train.columns if train[cols].isnull().sum() / train.shape[0] > 0.9
    ]
    cols_drop_test = [
        cols for cols in test.columns if test[cols].isnull().sum() / test.shape[0] > 0.9
    ]
    big_top_value_cols = [
        col
        for col in train.columns
        if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9
    ]
    big_top_value_cols_test = [
        col
        for col in test.columns
        if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9
    ]
    drop_cols = list(
        set(cols_drop_train + cols_drop_test + big_top_value_cols + big_top_value_cols_test)
    )
    drop_cols.remove("isFraud")

    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)

    del (
        cols_drop_test,
        cols_drop_train,
        big_top_value_cols,
        big_top_value_cols_test,
        drop_cols,
    )
    
    return train, test

def map_emails(df):
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 
          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 
          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 
          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 
          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

    us_emails = ['gmail', 'net', 'edu']

    #https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654
    for c in ['P_emaildomain', 'R_emaildomain']:
        df[c + '_bin'] = df[c].map(emails)
        
        df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])
        
        df[c + '_suffix'] = df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    return df


def map_transaction_amount(train, test):
    test['isFraud'] = 'test'
    temp = pd.concat([train, test], axis=0, sort=False)

    temp['Trans_min_mean'] = temp['TransactionAmt'] - temp['TransactionAmt'].mean()
    temp['Trans_min_std'] = temp['Trans_min_mean'] / temp['TransactionAmt'].std()

    temp['TransactionAmt_log'] = np.log(temp['TransactionAmt'])
    
    nice_vals = [0, 950, 949, 500]
    temp['TransactionAmt_cents'] = np.floor((temp['TransactionAmt'] % 1) * 1000)
    temp['TransactionAmt_cents_nice'] = temp['TransactionAmt_cents'].isin(nice_vals)
    
    train = temp[temp['isFraud'] != 'test']
    test = temp[temp['isFraud'] == 'test'].drop('isFraud', axis=1)

    return train, test

def encode_categorical_features(df_train, df_test):
    for f in df_train.drop('isFraud', axis=1).columns:
        if df_train[f].dtype=='object' or df_test[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df_train[f].values) + list(df_test[f].values))
            df_train[f] = lbl.transform(list(df_train[f].values))
            df_test[f] = lbl.transform(list(df_test[f].values))       
    
    return df_train, df_test


def drop_V_features(df):
    # applying the PCA is major PITA, for now just drop it
    mas_v = [c for c in df.columns if c.startswith('V')]
    df = df.drop(mas_v, axis=1)

    return df

def map_transaction_dt(df):
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    df["Date"] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

    df['_Weekdays'] = df['Date'].dt.dayofweek
    df['_Hours'] = df['Date'].dt.hour
    df['_Days'] = df['Date'].dt.day

    df.drop('Date', axis=1, inplace=True)

    return df

def id_split(dataframe):
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]

    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]

    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]

    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]

    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    dataframe['had_id'] = 1
    gc.collect()
    
    return dataframe

