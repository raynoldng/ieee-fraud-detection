import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

def reduce_mem_usage(df, verbose=True):
    # Still not sure why people are using this so much
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def load_data(sample=False, reduce_mem=True):
    # if sample is true, returns only 10 000  rows for training and test
    df_trans = pd.read_csv("../input/train_transaction.csv")
    df_test_trans = pd.read_csv("../input/test_transaction.csv")
    df_id = pd.read_csv("../input/train_identity.csv")
    df_test_id = pd.read_csv("../input/test_identity.csv")

    if sample:
        df_trans = df_trans.sample(n=10000)
        df_test_trans = df_test_trans.sample(n=10000)

    if reduce_mem:
        df_trans = reduce_mem_usage(df_trans)
        df_test_trans = reduce_mem_usage(df_test_trans)
        df_id = reduce_mem_usage(df_id)
        df_test_id = reduce_mem_usage(df_test_id)

    df_train = df_trans.merge(
        df_id, how="left", left_index=True, right_index=True, on="TransactionID"
    )
    df_test = df_test_trans.merge(
        df_test_id, how="left", left_index=True, right_index=True, on="TransactionID"
    )

    print(df_train.shape)
    print(df_test.shape)

    # y_train = df_train['isFraud'].copy()
    del df_trans, df_id, df_test_trans, df_test_id
    gc.collect()

    return df_train, df_test


def set_X_and_y(df_train):
    X_train = df_train.sort_values('TransactionDT').drop(['isFraud', 
                                                        'TransactionDT', 
                                                        #'Card_ID'
                                                        ],
                                                        axis=1)
    y_train = df_train.sort_values('TransactionDT')['isFraud'].astype(bool)

    return X_train, y_train