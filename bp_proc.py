'''
Разные функции
'''

from bp_const import *
from pnumpyscipy import *
import matplotlib.pyplot as plt

def get_region_name(country:str)->str:
    '''
    По стране определяем регион
    '''
    pass

def subplot_main_visualisation(num_flt:int, curr_flt:int, dat:np.ndarray, res:np.ndarray, llst:list):
    '''
    Общая функция для subplots визуализаций на основе pnumpyscipy.test_filters
    для кривых одинаковой длины
    В отличие от test_filters нет исходного варианта
    (num_flt, curr_flt, res, llst) = work_with_smoothing(x, y)  (int, int, np.ndarray, list):

    Матрица res
    # первый столбец (0) - номер кривой
    # второй столбец (1) - название кривой
    # третий столбец (2) - номер подварианта кривой
    # четвертый столбец (3) - кривая, данные
    # пятый столбец (4) - параметры подварианта кривой
    # шесто столбец (5) - х значения для кривой

    '''
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

if __name__ == "__main__":
    subplot_main_visualisation()