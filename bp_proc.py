'''
Разные функции
'''

from bp_const import *
from pnumpyscipy import *
import numpy as np
import matplotlib.pyplot as plt

def create_test_dat_for_subplot_main_visualisation(is_view=False)->list:
    '''
    Работа с вложенными списками, внутри которых дата-класс
    '''
    num_flt = 2
    ini_sub_var = 3
    len_x = 10; step_on_x = 10
    llst = []
    for i in range(num_flt):
        x_ = np.linspace(1, len_x, len_x)
        currlst = []
        l_p = str(i)
        for j in range(ini_sub_var):
            y_ = x_**2 + j*x_*2
            adds = ' ini ' if j == 0 else ''
            currlst.append(Line_(x=x_, y=y_, name= 'nm '+l_p, param = 'prm '+adds+' '+str(j),
                                 num = i, currnum=j))
        llst.append(currlst)
    return llst

def print_lst_lst_Line_(llst:list)->None:
    '''
    Печать вложенного списка, внутри которого дата-класс
    '''
    # print(inspect.currentframe().f_code.co_name)
    llen = len(llst)
    print('llen = ', llen)
    for i in range(llen):
        print('============= i = ' ,i)
        llen2 = len(llst[i])
        for j in range(llen2):
            print('j = ' ,j)
            print(llst[i][j])
            print()
    print(llst)


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
    sp = get_suplotsize(num_flt)
    plt.subplots( nrows = sp[0], ncols = sp[1], figsize=(15, 8))
    currflt = 0
    for i in range(num_flt): # перебор по всем фильтрам
        plt.subplot(sp[0],sp[1], i + 1)
        titl = llst[i][0].name
        plt.title(titl)
        for j in range(len(llst[i])): # перебор по всем подвариантам фильтров
            if llst[i][j].param != '':
                # Пустая строка означает, что фильтр не рассчитался
                plt.plot(llst[i][j].x, llst[i][j].y, label = llst[i][j].param)
            currflt += 1
            # https://jenyay.net/Matplotlib/LegendPosition
            plt.legend(loc = 'lower center', prop={'size': 8}) #  'lower right'  'best'
        plt.grid()
    plt.show()

if __name__ == "__main__":
    llst = create_test_dat_for_subplot_main_visualisation()
    # print_lst_lst_Line_(llst)
    subplot_main_visualisation(llst)