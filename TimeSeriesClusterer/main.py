import os
from loader import *
from imputer import *
from transformer import *
from regularizer import *
from clusterer import *
from itertools import product
from datetime import datetime as dt

DATASET_HOME = os.environ['DATASET_HOME']
DATE_COLUMN = None
VALUE_COLUMN = None
MIN_DATE = '2016'
MAX_DATE = None
LEFT_HINGE = 0
RIGHT_HINGE = None
RE_MATCH = '^ATM_[^.]*_ALL.csv$'

def main():
    t1 = dt.now()
    transformations = [None, 'roll_mean', 'weighted_roll_mean', 'diff']
    standardizations = [None, 'min_max','z']

    DF = load_csvs(path = DATASET_HOME, date_col = DATE_COLUMN, value_col = VALUE_COLUMN,
                   interpolate = True, min_date = MIN_DATE, max_date = MAX_DATE,
                   min_val = LEFT_HINGE, max_val = RIGHT_HINGE, file_match = RE_MATCH)

    print ('Data Load Time : ',dt.now()-t1)
    t2 = dt.now()
    for transformation,regularization in product(transformations,standardizations):
        t3 = dt.now()
        print ('*'*120)
        print ("Transformation ------> ",transformation)
        print ("Regularization ------> ", regularization)
        df = DF.copy()
        if transformation == 'roll_mean':
            df = df.apply(func = roll_mean, axis = 1)
        elif transformation == 'weighted_roll_mean':
            df = df.apply(func = weighted_roll_mean, axis = 1)
        elif transformation == 'diff':
            df = df.apply(func = difference, axis = 1)
        if regularization == 'min_max':
            df = df.apply(func = min_max, axis = 1)
        elif regularization == 'z':
            df = df.apply(func = z_transform, axis = 1)
        t4 = dt.now()
        model = KMeansHelper()
        model.fit(df)
        print ('All Fits : ',model.get_all_state())
        print ('Optimal Fit : ', model.get_optimal_model().maps)
        print ('Data Transformation Time :', t4-t3)
        print ('Clustering Time : ', dt.now()-t4)
        print ('Total Time : ', dt.now() - t3)
    print ('Total Clustering Process Time : ', dt.now()-t2)
    print ('Total Run Time : ',dt.now()-t1)

if __name__ == "__main__":
    main()
