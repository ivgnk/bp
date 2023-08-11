'''
Мои операции c pandas для bp-stats-review-2022-consolidated-dataset-panel-format.xlsx
'''
import pandas as pd
from bp_const import *
from copy import deepcopy

def extract_in_years(newdat: pd.DataFrame, limits: np.ndarray )->pd.DataFrame:
    '''
    Извлечение из pd.DataFrame строк в диапазоне заданных лет
    '''
    for i in range(limits.size):
        mmin = limits[i].begYear
        mmax = limits[i].endYear
        shrt_name = limits[i].shrt_name
        # print('i=', i, ' ', shrt_name)
        for index, row in newdat.iterrows():
            if  (row[index_moc] == shrt_name) and ((row["Year"]<mmin) or (row["Year"]>mmax)):
                newdat = newdat.drop( labels=[index],axis = 0, inplace = False)
                # print('drop ', row)
    # print(newdat)
    return newdat
