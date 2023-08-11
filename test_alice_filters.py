'''
Тестирование фильтров скользящего среднего,
рекомендованных Яндекс-Алисой и midjourney. Ничего не работает как надо
'''

from random import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def moving_average(arr, window_size):
    # Инициализируем массив для хранения скользящего среднего значения
    moving_avg = [0] * window_size

    # Проходим по каждому элементу в массиве
    for i in range(len(arr)):
        # Добавляем текущий элемент к скользящему среднему
        moving_avg[i % window_size] += arr[i]

        # Если текущий элемент последний, то очищаем среднее значение
        if i >= window_size - 1:
            moving_avg.pop(0)

    return moving_avg
def test_moving_average():
    # Пример использования функции для вычисления скользящего среднего на основе 5 элементов
    llen = 40
    arr =  [1.0, 2.0, 3.0, 4.0, 5.0] # [random()*10 for i in range(llen)]
    window_size = 5
    moving_avg_arr = moving_average(arr, window_size)
    print(moving_avg_arr)

def sliding_window_filter(sequence, window_length):
    if len(sequence) % 2 == 1:
        window_length -= 1
    # Создаем пустой список для хранения значений фильтра
    filter_values = []
    for i in range(window_length, len(sequence), 2):
        filter_values.append(sequence[i])
    result = np.mean(filter_values)
    print(type(result))
    return result

def test_sliding_window_filter():
    llen = 40
    arr =  np.array([random()*10 for i in range(llen)])
    window_size = 5
    print(type(arr))
    print(arr)
    sliding_window_filter_arr = sliding_window_filter(arr, window_size)
    print(sliding_window_filter_arr)
    plt.subplots(figsize=(15, 8))
    plt.plot(arr, label = 'ini')
    plt.plot(sliding_window_filter_arr, label = '5')
    plt.show()

def moving_average_midjourney(data, window_size):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        average = sum(window) / window_size
        moving_averages.append(average)
    return moving_averages

def moving_average_midjourney2(data, window_size, fill_left=None, fill_right=None):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        if len(window) < window_size:
            if fill_left is not None:
                window = [fill_left] * (window_size - len(window)) + window
            if fill_right is not None:
                window = window + [fill_right] * (window_size - len(window))
        average = sum(window) / window_size
        moving_averages.append(average)
    return moving_averages

def test_moving_average_midjourney():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = deepcopy(data)
    window_size = 3
    # result = moving_average_midjourney(data, window_size)
    result2 = moving_average_midjourney2(data, window_size, )
    plt.subplots(figsize=(15, 8))
    plt.plot(x, data, label = 'ini')
    print(len(x), len(result2))
    plt.plot(x, result2, label = 'v2 '+str(window_size))
    plt.legend()
    plt.show()
    print(result2)

if __name__ == "__main__":
    # test_moving_average()
    # test_sliding_window_filter()
    test_moving_average_midjourney()