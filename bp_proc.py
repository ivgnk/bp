'''
Разные функции
'''

from bp_const import *
from pnumpyscipy import *
import matplotlib.pyplot as plt

def get_region_name(country:str)->str:
    pass

def subplot_main_visualisation(num_flt:int):
    sp = get_suplotsize(num_flt)
    plt.subplots( nrows = sp[0], ncols = sp[1], figsize=(15, 8))
    currflt = 0
    for i in range(num_flt): # перебор по всем фильтрам
        plt.subplot(sp[0],sp[1], i + 1)
        plt.title(res[currflt,1])
        plt.plot(x, y, label='ini', linewidth=5)
        for j in range(llst[i]): # перебор по всем подвариантам фильтров
            if res[currflt,4] != '':
                # Пустая строка означает, что фильтр не рассчитался
                plt.plot(x, res[currflt,3], label = res[currflt,1]+' '+res[currflt,4])
            currflt += 1
            # https://jenyay.net/Matplotlib/LegendPosition
            plt.legend(loc = 'lower center', prop={'size': 8}) #  'lower right'  'best'
    plt.show()
