'''
Дополнительная информация и тестирование функций
'''

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

x = np.array([i for i in range(10)])
y = np.array([random() for i in range(10)])

res = work_with_smoothing(x, y)