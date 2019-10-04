import sys, os
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
sys.path.append(path + '/../')

import numpy as np
from sklearn import preprocessing

from fraud.tuning import optimize_hyper_parameters
import fraud.utils as utils

#############################################
# Feature Engineering
#############################################

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

def map_transaction_amount(df):
    df['Trans_min_mean'] = df['TransactionAmt'] - df['TransactionAmt'].mean()
    df['Trans_min_std'] = df['Trans_min_mean'] / df['TransactionAmt'].std()

    df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('std')

    df['TransactionAmt_log'] = np.log(df['TransactionAmt'])

    df['TransactionAmt_cents'] = df['TransactionAmt'] % 1

    return df


def encode_categorical_features(df_train, df_test):
    # NOTE this is the only feature engineering function that takes in both df_train 
    # df_test, this is because we want values across test and train datasets

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

if __name__ == "__main__":
    df_train, df_test = utils.load_data(sample=True)
    df_train = map_emails(df_train)
    df_test = map_emails(df_test)

    # df_train = drop_V_features(df_train) # this should be why the training is so slow
    # df_test = drop_V_features(df_test) # this should be why the training is so slow

    df_train = map_transaction_amount(df_train)
    df_test = map_transaction_amount(df_test)

    df_train, df_test = encode_categorical_features(df_train, df_test)
    
    X_train, y_train = utils.set_X_and_y(df_train)

    best = optimize_hyper_parameters(X_train, y_train, 8)
    print(best)