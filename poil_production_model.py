'''
My oil production model
'''

import numpy as np
from dataclasses import dataclass
import configparser
import os.path
import pprint as pp
import sys
import inspect

sec_param_name:list = ['a_TimeSectionParam', 'b_TimeSectionParam', 'c_TimeSectionParam', 'd_TimeSectionParam']
param_data_name = ['tm', 'tmc', 'per', 'amp']

view_ini = bool(1)

class OilProductionModel():
    def __init__(self, dat):
        if type(dat) == str:
            if os.path.exists(dat):
                # print('file exist')
                self.the_oil_producion_param = self.read_ini(dat)
                if view_ini: self.view_param()
            else:
                print('No file');  sys.exit(0)
        else: # экземпляр класса OilProductionModelParam
            print('No file name'); sys.exit(0)

    def create_empty_dict(self)->dict:
        OilProductionModelParam = dict()
        for s in sec_param_name:
            OilProductionModelParam[s] = dict()
            for s2 in param_data_name:
                OilProductionModelParam[s][s2] = 0.0
        return OilProductionModelParam

    def read_ini(self,fn:str):
        # https://habr.com/ru/articles/485236
        config = configparser.ConfigParser()  # создаём объекта парсера
        config.read(fn)
        if view_ini: self.view_config(config)
        return self.convert_config2OilProductionModelParam(config)

    def view_config(self,config):
        print('\n',inspect.currentframe().f_code.co_name)
        for i,s in enumerate(sec_param_name):
            print(f'[{sec_param_name[i]}]')
            for j, s2 in enumerate(param_data_name):
                print(f'{param_data_name[j]} = {config[s][s2]}')

    def convert_config2OilProductionModelParam(self,config):
        #-(1) ini
        OilProductionModelParam = self.create_empty_dict()
        for i,s in enumerate(sec_param_name):
            for j, s2 in enumerate(param_data_name):
                OilProductionModelParam[s][s2] = config[s][s2]
        return OilProductionModelParam


    def view_param(self):
        print('\n',inspect.currentframe().f_code.co_name)
        pp.pp(self.the_oil_producion_param)


if __name__ == "__main__":
    the_OPM =OilProductionModel('OilProductionModelParam.ini')


