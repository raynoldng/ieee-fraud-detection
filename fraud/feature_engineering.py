import pandas as pd
import datetime
import numpy as np
from sklearn import preprocessing

#############################################
# Feature Engineering
# TODO time delta mapping
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
    # TODO this is causing the tuning to fail
    test['isFraud'] = 'test'
    temp = pd.concat([train, test], axis=0, sort=False)

    temp['Trans_min_mean'] = temp['TransactionAmt'] - temp['TransactionAmt'].mean()
    temp['Trans_min_std'] = temp['Trans_min_mean'] / temp['TransactionAmt'].std()

    temp['TransactionAmt_to_mean_card1'] = temp['TransactionAmt'] / temp.groupby(['card1'])['TransactionAmt'].transform('mean')
    temp['TransactionAmt_to_mean_card4'] = temp['TransactionAmt'] / temp.groupby(['card4'])['TransactionAmt'].transform('mean')
    temp['TransactionAmt_to_std_card1'] = temp['TransactionAmt'] / temp.groupby(['card1'])['TransactionAmt'].transform('std')
    temp['TransactionAmt_to_std_card4'] = temp['TransactionAmt'] / temp.groupby(['card4'])['TransactionAmt'].transform('std')

    temp['TransactionAmt_log'] = np.log(temp['TransactionAmt'])

    temp['TransactionAmt_cents'] = temp['TransactionAmt'] % 1

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
