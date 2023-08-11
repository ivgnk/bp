'''
Работа с функциями obspy
'''

from obspy import read
import numpy as np
from bp_const import *
from pnumpyscipy import *
from obspy import read


# st = read()  # load example seismogram
# print(len(st))
#
# mra = get_random_arr(123)
#
# st_list = list(st)
# print(type(st))
# print(st_list)
# st.filter(type='highpass', freq=3.0)
# st = st.select(component='Z')
# st.plot()

def view_param():
    '''
    Просмотр параметров файла сейсмоданных
    '''
    st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
    print('st = ',st)
    print('len(st) = ',len(st))
    print('--------')
    tr = st[0]  # assign first and only trace to new variable
    print(tr.stats)
    tr_filt = tr.copy()
    print('len(tr) = ',len(tr))
    print('tr = ', tr)
    print('--------')
    print(type(st),' ',type(tr),' ',type(tr.data),type(tr_filt))

if __name__ == "__main__":
    view_param()