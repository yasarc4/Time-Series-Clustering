import os
import pandas as pd
import re
import ntpath
from imputer import *

def load_csvs(path,date_col = None, value_col = None, interpolate = True, file_match = '^[A-Za-z0-9_ ]*.csv$',
              min_date = None, max_date = None, min_val = None, max_val = None):
    if date_col == None and value_col == None:
        use_cols = None
    else:
        use_cols = [date_col, value_col]
    all_files = [path+i for i in os.listdir(path)  if re.match(file_match,i)]
    df = load_csv(all_files[0], use_cols = use_cols,interpolate = interpolate,
                  min_date = min_date, max_date = max_date, min_val = min_val, max_val = max_val)
    for f in all_files[1:]:
        df_ = load_csv(f,use_cols = use_cols,interpolate = interpolate,
                      min_date = min_date, max_date = max_date, min_val = min_val, max_val = max_val)
        df = pd.merge(df, df_, how = 'outer', left_index=True, right_index=True)
    df = df.transpose()
    if (interpolate == True) and (pd.isnull(df).sum().sum()>0):
        df = impute(df, method = 'forward_backward')
    return df

def get_basename(path):
    base = ntpath.basename(path)
    base = base[:base.find('.')]
    return base

def load_csv(path,use_cols = None, interpolate = True,
              min_date = None, max_date = None, min_val = None, max_val = None):
    if use_cols == None:
        df = pd.read_csv(path, index_col = 0)
    else:
        df = pd.read_csv(path, usecols=use_cols,index_col=0)
    df = df[~df.index.duplicated(keep='first')]
    colname = get_basename(path)
    df.columns = [colname]
    df = clean_df(df,min_date=min_date, max_date=max_date, min_val = min_val, max_val = max_val, colname = colname)
    if interpolate==True:
        df[colname] = df[colname].interpolate()
    return df

def clean_df(df,min_date=None, max_date=None, min_val = None, max_val = None, colname = None):
    if colname==None:
        colname = df.columns[0]
    if min_date!=None:
        df = df.loc[df.index>=min_date]
    if max_date!=None:
        df = df.loc[df.index<=max_date]

    if min_val!=None:
        if callable(min_val):
            min_val = min_val(df[colname])
        df.loc[df[colname]<min_val] = min_val
    if max_val!=None:
        if callable(max_val):
            max_val = max_val(df[colname])
        df.loc[df[colname]>max_val] = max_val
    return df
