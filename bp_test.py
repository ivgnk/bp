'''
Дополнительная информация и тестирование функций
Работа с Git в PyCharm. Без терминалов и головной боли.
https://www.youtube.com/watch?v=9VKKZNqzPcE
'''

from dataclasses import dataclass
from bp_const import *
from pnumpyscipy import *
from  random import *
# Данные
# Старые
# https://www.bp.com/en/global/corporate/energy-economics/statistical-review-of-world-energy.html
# https://www.bp.com/en/global/corporate/energy-economics/statistical-review-of-world-energy/downloads.html
# Новый с 2023 года
# https://www.energyinst.org/exploring-energy/statistical-review

# Работа с Git в PyCharm  https://www.youtube.com/watch?v=9VKKZNqzPcE
# Слить ветку master и main https://ru.stackoverflow.com/questions/1197561/
# я удалил ветку 'master' из репозитория на github, затем переименовал main в master,
# после этого повторно запушил проект на github. При запросе подтвердил rebase.


# Графические интерфейсы
# https://www.youtube.com/@zproger
# CustomTkinter https://www.youtube.com/watch?v=RKHBcOiViDo
# DearPyGui https://www.youtube.com/watch?v=Fkpr0au59aU

import matplotlib.pyplot as plt
import pandas as pd
import math
# df = pd.DataFrame({'c1': [10, 11, 0, 12], 'c2': [100, 110, 120, 0]})
# print(df); print()
# for index, row in df.iterrows():
#     if row['c1']==0:
#         row['c2'] = 1000
#     else:
#         row['c2'] = -1000
# print(df); print()

# S = math.nan
# if S:
#     print(True)
# else:
#     print(False)

# keys = list(main_oil_countries.keys())
# print(keys)
# values = list(main_oil_countries.values())
# print(values)
# ll = len(main_oil_countries)
# print(ll)
# for i in range(ll):
#     print(i,' ',keys[i],' ', values[i])

# x = [i for i in range(10)]
# x1= [i for i in range(2,12)]
# y = [i*2 for i in range(10)]
# y1= [i*3 for i in range(2,12)]
# plt.plot(x,y)
# plt.plot(x1,y1)
# plt.show()

# main_oil_countries2 = np.arange(3, dtype=CountryYears)
# print(type(main_oil_countries2))

# main_oil_countries2 = create_main_oil_countries2()
# print(type(main_oil_countries2))
# print(main_oil_countries2.size)

def test_dict():
    '''
    Разные операции со словарями
    '''
    x = [i for i in range(10)]; x1= [i for i in range(2,12)]; y = [i*2 for i in range(10)]; y1= [i*3 for i in range(2,12)]
    the_dict = {'x':x, 'x1':x1, 'y':y, 'y1':y1}

    k = ['x', 'x1', 'y', 'y1']; v = [x, x1, y, y1]
    the_dict2 = dict(zip(k,v))

    print(the_dict);   print(the_dict2)
    print('len(the_dict) = ', len(the_dict))

    # 4 способа перебора словаря в Python
    # https://python-lab.ru/4-sposoba-perebora-slovarya-v-python
    # https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
    print('\n var 1')
    i = 0
    for itms in the_dict.items():
        print(i,'  ',itms, '  ', )
        i += 1

    print('\n var 2')
    i = 0
    for key, value in the_dict.items():
        print(i,'  key = ',key, '    value = ', value, ' ')
        i += 1

    print('\n var 3')
    i = 0
    for key in the_dict.keys():
        print(i,'  key = ',key,'   the_dict[key] =',the_dict[key])
        i += 1

    print('\n var 4')
    i = 0
    for value in the_dict.values():
        print(i,'  value = ',value)
        i += 1


# ----------- dataclass в различных структурах данных

def test_dict_with_dataclass():
    '''
    Разные операции со словарями, содержащими dataclass
    '''
    print('\n base')
    curr_dict = dict()
    n = 5
    for i in range(n):
        # curr_dict[i] = i*10
        x1 = np.array([i for i in range(10)])
        y1 = np.array([i*2 for i in range(10)])
        s = str(i)
        curr_dict[i] = Line_(x=x1, y=y1, name='nm'+s, param='prm'+s, num=i, currnum = i)

    print(curr_dict)

    print('\n cycle for')
    for i in range(n):
        print('i = ',i,'   key = ',i,'  value = ', curr_dict[i])


def test_nested_lists():
    '''
    Тестирование вложенных списков
class Line_:
    x:np.ndarray
    y:np.ndarray
    name:str  # название линии (фильтра)
    param:str # параметры линии (фильтра)
    num:int   # номер линии (фильтра)
    currnum:int # номер подварианта линии (фильтра)
    '''
    llst = []
    n = 5
    for i in range(n):
        llst.append([])
        for j in range(3):
            # print(i, 'before ',j,' ',len(llst[i]))
            llst[i].append(j)
            # print(i, 'after ',j,' ',len(llst[i]))
            pass
    print(llst)

if __name__ == "__main__":
    test_nested_lists()
    # test_dict_with_dataclass()