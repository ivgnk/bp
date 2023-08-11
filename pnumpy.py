'''
Работа с функциями numpy
'''


import numpy as np
from functools import reduce

def test_np_sum():
    n1 = np.array([1, 2, 3, 4])
    n2 = np.array([2, 3, 4, 5])
    llst = [n1, n2]
    print('llst = ',llst)
    a = np.sum(llst, axis=0)
    print(a, ' ', type(a))
    a = np.sum([1, 2, 3, 4], axis=0)
    print(a, ' ', type(a))

def test_map_zip_filter_reduce():
    '''
    numpy аналоги встроенных функций map, zip, filter, reduce
    map - применение функции к каждому элементу последовательности
    zip - на каждой итерации возвращает кортеж, содержащий элементы
    последовательностей на одинаковом смещении
    filter - проверка элементов последовательности
    reduce - применяет фукнцию к парам элементов и накапливает результат
    в функцияю обр. вызова подаются 2 элемента:
    предыдущий результат и текущий элемент
    '''
    # -------(1)------- map1
    print('\n map1')
    def func(elem):
        return elem+10
    arr = [1, 2, 3, 4, 5]
    print(list(map(func, arr)))
    np_arr = np.array(arr)
    np_arr2 =np_arr+10
    print(np_arr2)

    # -------(1)------- map2
    print('\n map2 - суммирование элементов на одинакоой позиции')
    def func2(e1, e2, e3):
        return e1+e2+e3
    arr1 = [1, 2, 3, 4, 5];           nparr1 = np.array(arr1)
    arr2 = [10, 20, 30, 40, 50];      nparr2 = np.array(arr2)
    arr3 = [100, 200, 300, 400, 500]; nparr3 = np.array(arr3)
    print(list(map(func2, arr1, arr2, arr3)))
    print(nparr1+nparr2+nparr3)

    # -------(1)------- map3
    # stackoverflow.com/questions/14916407/how-do-i-stack-vectors-of-different-lengths-in-numpy
    print('\n map3 - суммирование элементов на одинакоой позиции \n при различном числе элементов')
    def func2(e1, e2, e3):
        return e1+e2+e3
    arr1 = [1, 2, 3];           nparr1 = np.array(arr1)
    arr2 = [10, 20];      nparr2 = np.array(arr2)
    arr3 = [100, 200, 300, 400, 500]; nparr3 = np.array(arr3)
    print(list(map(func2, arr1, arr2, arr3)))
    print('для разной длины numpy не работает')

    # -------(2)------- zip1
    print('\n zip1 - на каждой итерации возвращает кортеж, \n содержащий элементы  последовательностей на одинаковом смещении')
    arr1 = [1, 2, 3, 4, 5];           nparr1 = np.array(arr1)
    arr2 = [10, 20, 30, 40, 50];      nparr2 = np.array(arr2)
    arr3 = [100, 200, 300, 400, 500]; nparr3 = np.array(arr3)
    print(list(zip(arr1, arr2, arr3)))
    print(list(zip(nparr1, nparr2, nparr3)))

    # -------(2)------- zip2
    print('\nzip2 - на каждой итерации возвращает кортеж, \n содержащий элементы последовательностей на одинаковом смещении')
    print('последовательности разной длины')
    print('zip2 native')
    arr1 = [1, 2, 3, 4, 5];           nparr1 = np.array(arr1)
    arr2 = [10, 20, 30, 40];      nparr2 = np.array(arr2)
    arr3 = [100, 200, 300]; nparr3 = np.array(arr3)
    print(list(zip(arr1, arr2, arr3)))
    print(list(zip(nparr1, nparr2, nparr3)))

    print('zip2 c numpy')
    llen = min(nparr1.size, nparr2.size, nparr3.size)
    llst = []
    for i in range(llen):
        llst += [(nparr1[i], nparr2[i], nparr3[i])]
    print(llst)

    # -------(3)------- filter1
    print('\n filter1 - проверка элементов последовательности')
    print('filter1 native')
    llst = [1, 0, None, [], 2]
    print(list(filter(None, llst)))
    print('filter1 c numpy')
    a = np.array(llst, dtype=object)
    d = np.array([])
    b =  np.logical_and(a!=None,a!=0)  #, a!=d
    c = a[ b ]
    print(d)
    print('a = ', a)
    print('b = ', b)
    print('c = ', c)

    # -------(3)------- filter2
    print('\n filter2 - проверка элементов последовательности')
    print('filter2 с помощью генераторов списков')
    a = [i for i in llst if i]
    print('a = ', a)


if __name__ == "__main__":
    # test_np_sum()
    test_map_zip_filter_reduce()