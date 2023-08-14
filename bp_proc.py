'''
Разные функции
'''

from bp_const import *
from pnumpyscipy import *
import numpy as np
import matplotlib.pyplot as plt

def get_region_name(country:str)->str:
    '''
    По стране определяем регион
    '''
    pass

def create_test_dat_for_subplot_main_visualisation(is_view=False)->list:
    num_flt = 2
    ini_sub_var = 3
    len_x = 50; step_on_x = 10
    llst = []
    for i in range(num_flt):
        x_ = np.linspace(1, len_x, len_x)
        currlst = []
        l_p = str(i)
        for j in range(ini_sub_var):
            y_ = x_**2 + j*x_*2
            currlst.append([])
            currlst[j] = Line_(x=x_, y=y_, name= 'nm '+l_p, param = 'prm '+l_p, num = i, currnum=j)
        llst.append(currlst)
        # plt.plot(x,y)
    # plt.grid()
    # plt.show()
    if is_view:
        llen = len(llst)
        print('llen = ', llen)
        for i in range(llen):
            print('============= i = ' ,i)
            llen2 = len(llst[i])
            for j in range(llen2):
                print('j = ' ,j)
                print(llst[i][j])
                print()
    return llst

def subplot_main_visualisation(llst:list):
    '''
    Общая функция для subplots визуализаций на основе pnumpyscipy.test_filters
    для кривых произвольной длины
    llst - список списков
    Каждый llst[i] - отдельный фильтр, который содержит список
    экземпляров дата-класса bp.const.Line_

class Line_:
    x:np.ndarray
    y:np.ndarray
    name:str  # название линии (фильтра)
    param:str # пароаметры линии (фильтра)
    num:int   # номер линии (фильтра)
    currnum:int # номер подварианта линии (фильтра)
    '''

    num_flt = len(llst)
    # sp = get_suplotsize(num_flt)
    # plt.subplots( nrows = sp[0], ncols = sp[1], figsize=(15, 8))
    # currflt = 0
    # for i in range(num_flt): # перебор по всем фильтрам
    #     plt.subplot(sp[0],sp[1], i + 1)
    #     plt.title(res[currflt,1])
    #     plt.plot(x, y, label='ini', linewidth=5)
    #     for j in range(llst[i]): # перебор по всем подвариантам фильтров
    #         if res[currflt,4] != '':
    #             # Пустая строка означает, что фильтр не рассчитался
    #             plt.plot(x, res[currflt,3], label = res[currflt,1]+' '+res[currflt,4])
    #         currflt += 1
    #         # https://jenyay.net/Matplotlib/LegendPosition
    #         plt.legend(loc = 'lower center', prop={'size': 8}) #  'lower right'  'best'
    # plt.show()
    pass

if __name__ == "__main__":
    create_test_dat_for_subplot_main_visualisation(True)
    # subplot_main_visualisation()