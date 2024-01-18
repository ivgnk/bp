'''
Oil production model

(C) 2024 Ivan Genik, Perm, Russia
Released under GNU Public License (GPL)
email: igenik@rambler.ru
'''

import numpy as np
import matplotlib.pyplot as plt
import math

import configparser
import os.path
import pprint as pp
import sys
import inspect
from pathlib import Path
import pmath_trig as mt

# - Интервалы времени добычи
# в самом первом значении добыча = 0
# в самом последнем значении добыча = 0

# a_TimeSectionParam - рост добычи,
# b_TimeSectionParam - "плато" добычи
# c_TimeSectionParam - быстрый спад добычи
# d_TimeSectionParam - медленный спад добычи
# добыча растет от 0 до 1
# - Параметры каждого интервала времени
# -- временная зависимость состоит из комбинации
# -- прямой с тем или иным наклоном и синусоиды/косинусоиды
# tm - время периода - кварталы
# tmc - наклон кривой добычи
# per - период колебания в кварталах
# amp - амплитуда колебаний в кварталах
sec_param_name: list = ['a_TimeSectionParam', 'b_TimeSectionParam',
                        'c_TimeSectionParam', 'd_TimeSectionParam', 'curve_param']
param_data_name = ['tm', 'tmc', 'per', 'amp']
prm_dt_nm_add1 = ['beg', 'max']

view_ini = bool(1)
view_func_name = bool(0)

# --- тестовые параметры и файл
test_OPM = {'a_TimeSectionParam': {'tm':  20, 'tmc':  26.0, 'per':  6.0, 'amp':  1.0},
            'b_TimeSectionParam': {'tm':  70, 'tmc':   0.3, 'per': 16.0, 'amp': 11.0},
            'c_TimeSectionParam': {'tm':  90, 'tmc':  -5.4, 'per': 26.0, 'amp': 21.0},
            'd_TimeSectionParam': {'tm': 120, 'tmc':  -0.2, 'per': 36.0, 'amp': 31.0},
            'curve_param': {'beg':0.2, 'max':1.0}
            }

OPM_ini_fn = 'OilProductionModelParam.ini'
test_OPM_ini_fn = 'test_'+OPM_ini_fn


# - Непосредственно определение класса
class OilProductionModel:
    def __init__(self, dat, is_view: bool = False):
        if type(dat) == str:
            if os.path.exists(dat):
                # print('file exist')
                self.the_oil_producion_param = self.read_ini(dat)
                if view_ini: self.view_param()
            else:
                print('No file');  sys.exit(0)
        else:  # экземпляр класса OilProductionModelParam
            print('No file name'); sys.exit(0)
        # https://ru.stackoverflow.com/questions/535318/Текущая-директория-в-python
        self.curr_dir = str(Path.cwd())

        self.npoint: int = self.calc_npoint_OPM(is_view)
        self.arg: np.ndarray = np.linspace(0, self.npoint-1, self.npoint) # аргументы, время, номер квартала
        self.data: np.ndarray = np.zeros(self.npoint) # функция от аргумента, величина добычи
        self.sect_param: np.ndarray = self.calc_OPM_sect_param()
        self.calc_OilProductionModel()
        if is_view: self.visu_OilProductionModel()
        self.write_test_ini(test_OPM_ini_fn)
        self.calc_OilProductionModel()

    def calc_npoint_OPM(self, is_view: bool = False):
        if view_func_name: print('\n'+inspect.currentframe().f_code.co_name)
        npoints: int = 0
        for i, s in enumerate(sec_param_name):
            if i<=3:
                npoints += self.the_oil_producion_param[s]['tm']
        if is_view:
            print(f'\nВ модели {npoints=} кварталов')
            print(f'т.е {npoints/4=} лет')
        return npoints

    def create_empty_dict(self) -> dict:
        oil_production_model_param = dict()
        for i,s in enumerate(sec_param_name):
            oil_production_model_param[s] = dict()
            if i<=3:
                for s2 in param_data_name:
                    oil_production_model_param[s][s2] = 0.0
            else:
                for s2 in prm_dt_nm_add1:
                    oil_production_model_param[s][s2] = 0.0


        return oil_production_model_param

    def read_ini(self, fn: str):
        # https://habr.com/ru/articles/485236
        print('\n'+inspect.currentframe().f_code.co_name)
        config = configparser.ConfigParser()  # создаём объекта парсера
        config.read(fn)
        if view_ini: self.view_config(config)
        return self.convert_config_to_OilProductionModelParam(config)

    def write_test_ini(self, thetest_OPM_ini_fn:str, thetest_OPM:dict=test_OPM):
        if view_func_name: print('\n'+inspect.currentframe().f_code.co_name)
        config = configparser.ConfigParser()  # создаём объекта парсера
        # https://docs.python.org/3/library/configparser.html
        for i, s in enumerate(sec_param_name):
            config[s] = thetest_OPM[s]
        # new_OPM_ini_fn = '\\'.join([self.curr_dir, thetest_OPM_ini_fn])
        with open(thetest_OPM_ini_fn, 'w') as configfile:
            config.write(configfile)


    def view_config(self, config):
        if view_func_name: print('\n'+inspect.currentframe().f_code.co_name)
        for i, s in enumerate(sec_param_name):
            print(f'[{sec_param_name[i]}]')
            if i<=3:
                for j, s2 in enumerate(param_data_name):
                    print(f'{param_data_name[j]} = {config[s][s2]}')
            else:
                for j, s2 in enumerate(prm_dt_nm_add1):
                    print(f'{prm_dt_nm_add1[j]} = {config[s][s2]}')


    def convert_config_to_OilProductionModelParam(self, config):
        # -(1) ini
        if view_func_name: print('\n', inspect.currentframe().f_code.co_name)
        oil_production_model_param = self.create_empty_dict()
        for i, s in enumerate(sec_param_name):
            if i<=3:
                for j, s2 in enumerate(param_data_name):
                    dat_ = config[s][s2]
                    if j == 0:
                        oil_production_model_param[s][s2] = int(dat_)
                    else:
                        oil_production_model_param[s][s2] = float(dat_)
            else:
                for j, s2 in enumerate(prm_dt_nm_add1):
                    dat_ = config[s][s2].strip()
                    oil_production_model_param[s][s2] = float(dat_)
        print('------------')
        pp.pp(oil_production_model_param)
        print('------------')
        return oil_production_model_param

    def view_param(self):
        if view_func_name: print('\n'+inspect.currentframe().f_code.co_name)
        pp.pp(self.the_oil_producion_param)

    def calc_OPM_sect_param(self):
        sect_param = (-20)+np.zeros(self.npoint, dtype=np.int_) # заготовка массива
        curr_param = 0
        curr_gran = self.the_oil_producion_param[sec_param_name[curr_param]]['tm']
        for i in range(self.npoint):
            if i< curr_gran:
                sect_param[i] = curr_param
            else:
                curr_param += 1
                curr_gran += self.the_oil_producion_param[sec_param_name[curr_param]]['tm']
                sect_param[i] = curr_param
        # print(sect_param)
        return sect_param

    def Normir_OilProductionModel(self):
        max_= np.max(self.data)
        norm_data = self.the_oil_producion_param['curve_param']['max']
        self.data = self.data / max_ * norm_data
        # for i in range(self.npoint):
        #     self.data[i] = self.data[i]/max_*norm_data

    def calc_OilProductionModel(self):
        for i in range(self.npoint):
            if i==0:
                self.data[i] = self.the_oil_producion_param['curve_param']['beg']
            else:
                name_ = sec_param_name[self.sect_param[i]]
                tmc = self.the_oil_producion_param[name_]['tmc']
                per = self.the_oil_producion_param[name_]['per']
                amp = self.the_oil_producion_param[name_]['amp']
                self.data[i] = self.data[i-1]+tmc*(self.arg[i]-self.arg[i-1]) # +amp*math.sin()
            # print(f'{i:3}  {self.arg[i]:6}   {self.data[i]}')
        self.Normir_OilProductionModel()

    def visu_OilProductionModel(self):
        plt.plot(self.arg, self.data)
        plt.grid()
        plt.show()



if __name__ == "__main__":
    the_OPM = OilProductionModel(OPM_ini_fn, is_view=True)  # 100 лет по 4 квартала
