import pandas as pd

def impute(df, method = 'forward_backward'):
    if method == 'forward_backward':
        df = forward_impute(df)
        df = backward_impute(df)
    elif method == 'mean':
        df = mean_impute(df)
    return df

def forward_impute(df):
    cols_to_impute = zip(df.columns[1:],df.columns)
    for target,col in cols_to_impute:
        null_members = pd.isnull(df[target])
        try:
            df[target].loc[null_members]=df[col].loc[null_members]
        except:
            print ('NULL MEMBERS : ', null_members[:5], null_members.shape, df.shape)
            print ('Forward Impute Failed : Target = ',target)
    return df

def backward_impute(df):
    cols_to_impute = zip(df.columns[-2::-1],df.columns[::-1])
    for target,col in cols_to_impute:
        null_members = pd.isnull(df[target])
        try:
            df[target].loc[null_members]=df[col].loc[null_members]
        except:
            print ('NULL MEMBERS : ', null_members[:5], null_members.shape, df.shape)
            print ('Backward Impute Failed : Target = ',target)
    return df

def mean_impute(df):
    means = df.mean(axis = 1)
    for col in df.columns:
        null_members = pd.isnull(df[col])
        df[col].loc[null_members] = means.loc[null_members]
