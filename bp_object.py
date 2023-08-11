'''
Основной объект программы bp_main_analysis:
данные и их обработка
'''

import os
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bp_const import *
from ppandas import *


class BpDat:
    bp_DataFrame: pd.DataFrame  # полный набор данных
    bp_oil_price: pd.DataFrame  # цены на нефть

    curr_dir = ''
    curr_dat_dir = ''
    curr_res_dir = ''

    def __init__(self):
        print('bp_dat.__init__')
        self.def_dir()
        self.the_input_panel_data()
        self.panel_frame_rebuilding()

        self.the_output_panel_data()

        # т.к. в panel data данные только по странам
        self.the_input_oil_price_sheet()
        self.price_frame_shrinking(bool(0))  # удаление пустых строк и столбцов в bp_oil_price
        print('end __init__')

    def def_dir(self):
        self.curr_dir = os.getcwd()
        self.curr_dat_dir = "\\".join([self.curr_dir, dat_dir])
        self.curr_res_dir = "\\".join([self.curr_dir, res_dir])
        # print(self.cur_dat_dir, ' ', self.curr_res_dir)

    def the_input_panel_data(self):
        # panel_xls_fname = 'bp-stats-review-2022-consolidated-dataset-panel-format.xlsx'
        # main_xls_fname = 'bp-stats-review-2022-all-data.xlsx'
        bp_xlsx = pd.ExcelFile("\\".join([self.curr_dat_dir, panel_xls_fname]))
        # bp_xlsx = pd.ExcelFile("dat/bp-stats-review-2022-consolidated-dataset-panel-format.xlsx")
        # print(bp_xlsx.sheet_names)
        self.bp_DataFrame = bp_xlsx.parse()

    def the_output_panel_data(self):
        # 2023_Python и анализ данных
        # Первичная обработка данных с применением pandas, NumPy и Jupiter.pdf
        # стр.205
        fname = "\\".join([self.curr_res_dir, my_panel_xls_fname])
        self.bp_DataFrame.to_excel(fname, index=False)
        pass

    def panel_frame_add_col_opecplus(self, is_view=False):
        # Вставить колонку и задать признак OPEC+
        if is_view:
            self.print_row_col(self.bp_DataFrame, 'Перед '+Col_Opec_plus)
        self.bp_DataFrame[Col_Opec_plus] = 0

        for index, row in self.bp_DataFrame.iterrows():
            if (row['Country'] in opecplus_list) or (row['Country'] in opec_list):
                self.bp_DataFrame.loc[index, Col_Opec_plus] = 1
        if is_view:
            self.print_row_col(self.bp_DataFrame, 'После '+Col_Opec_plus, True)

    def panel_frame_restore_region(self):
        # восстановить значение Region в случае пропуска
        print('self.panel_frame_restore_region')
        for index, row in self.bp_DataFrame.iterrows():
            if type(row['Region']) != str:
                for x in spec_region_list:
                    if x in row['Country']:
                        self.bp_DataFrame.loc[index, 'Region'] = x

    # отдельно 2 трудных случая 'Other', выпадающих из общей закономерности
        for index, row in self.bp_DataFrame.iterrows():
            if type(row['Region']) != str:
                if row['Country'] in spec_region_list2:
                    self.bp_DataFrame.loc[index, 'Region'] = 'S. & Cent. America'

    # отдельно 2 трудных случая 'Total', выпадающих из общей закономерности
        for index, row in self.bp_DataFrame.iterrows():
            if type(row['Region']) != str:
                if row['Country'] == 'Total EU':
                    self.bp_DataFrame.loc[index, 'Region'] = 'Europe'
                if row['Country'] == 'Total Central America':
                    self.bp_DataFrame.loc[index, 'Region'] = 'S. & Cent. America'

    def panel_frame_rebuilding(self):
        self.panel_frame_add_col_opecplus()
        self.panel_frame_restore_region()

    def the_input_oil_price_sheet(self):
        # т.к. в panel data данные только по странам
        # bp_xlsx = pd.ExcelFile("\\".join([self.curr_dat_dir, panel_xls_fname]))
        bp_xlsx = pd.ExcelFile("\\".join([self.curr_dat_dir, main_xls_fname]))
        # print(bp_xlsx.sheet_names)
        self.bp_oil_price = bp_xlsx.parse(sheet_name=main_xls_oil_price_sheet)
        # print('bp_oil_price')
        # print(self.bp_oil_price)

    def print_row_col(self, dat: pd.DataFrame, title: str, view_head_tile: bool = False):
        print(title)
        print('число строк    = ',  dat.shape[0], ' ', dat.axes[0])
        print('число столбцов = ',  dat.shape[1], ' ', dat.axes[1])
        if view_head_tile:
            print(dat.head(7)); print(dat.tail(7))
        print()

    def price_frame_shrinking(self, is_view: bool = False):
        if is_view: self.print_row_col(self.bp_oil_price, 'Перед price_frame_rebuilding')
        bp: pd.DataFrame = self.bp_oil_price.drop(index=[165, 166, 167, 168])  # удаление строк
        bp = bp.drop(index=[164])  # удаление строки, ее не удается удалить сразу
        bp = bp.drop(index=[0, 1, 2])  # удаление строк
        cols = list(range(3, 12))
        bp.drop(bp.columns[cols], axis=1, inplace=True)
        bp.columns = ['Year', '$ money of the day', '$ 2021']
        rows = bp.shape[0]

        bp = bp.reset_index()  # сбрасыввем индекс, но старый индекс отсается первой колонкой
        bp.drop(bp.columns[0], axis=1, inplace=True)  # удаляем колонку старого индекса
        self.bp_oil_price = bp
        if is_view:
            self.print_row_col(self.bp_oil_price, 'После price_frame_rebuilding', True)

    def the_work(self):
        print('bp_dat.the_work')

    def the_visualisation(self, is_view=False):
        if is_view:
            print('the_visualisation')
            self.the_oil_price_the_visualisation()
            self.the_oil_production_in_year_visualisation(True)

    def the_oil_production_in_year_visualisation(self, is_view=False):
        y0 = years_prod_visu[0]
        y1 = years_prod_visu[1]
        y1965 = self.extract_oilprod_kbd(y0, clm_nm)
        yyear = self.extract_oilprod_kbd(y1, clm_nm)
        xtick_ = np.arange(len(country_prod_visu))  # [0 1 2]  числовые метки оси x
        width = 0.2  # ширина столбцов

        fig, ax = plt.subplots(figsize=(15, 8))
        plt.grid()
        # создаем столбцы слева (x - width/2) и справа (x + width/2)
        # от каждой метки x
        # label – названия элементов легенды
        rects1 = ax.bar(xtick_ - width / 2, y1965, width, label=str(y0))
        rects2 = ax.bar(xtick_ + width / 2, yyear, width, label=str(y1))

        ax.set_title(chrt_ttl)
        ax.set_xticks(xtick_)  # устанавливаем метки оси х
        ax.set_xticklabels(country_prod_visu)  # устанавливаем надписи меток оси х
        ax.legend()  # отображаем легенду
        plt.show()

    def extract_oilprod_kbd(self, year_: int, clm_nm: str):
        len_ = len(country_prod_visu)
        mass = [None]*len_
        # print('year_ = ', year_)
        for rowIndex, row in self.bp_DataFrame.iterrows():  # iterate over rows
            if (row['Country'] in country_prod_visu) and (row['Year'] == year_):
                for j in range(len_):
                    if row['Country'] == country_prod_visu[j]:
                        mass[j] = row[clm_nm]
        # print(mass)
        return mass

    def the_oil_price_the_visualisation(self, is_view=False):
        fig, ax = plt.subplots()
        plt.grid()  # создание сетки
        ax.set_title('Цены на нефть, доллары')
        x = self.bp_oil_price[theYear]
        y = self.bp_oil_price[themotd]
        y1 = self.bp_oil_price[thed2021]
        ax.plot(x, y, label=themotd)
        ax.plot(x, y1, label=thed2021)
        ax.legend()
        plt.show()
        ###################

    def the_end(self):
        print('Нормальное завершение')

    def work_with_country(self, country_: str):
        print('work_with ' + country_)
        # SA_Series = self.bp_DataFrame["oilprod_kbd"]
        dat = self.bp_DataFrame
        temp = dat[["Country", "Year", clm_nm]]  # clm_nm == "oilprod_kbd"
        C_Series=temp[temp["Country"].str.contains(country_)]
        x = np.array(C_Series["Year"].values.tolist())
        y = np.array(C_Series[clm_nm].values.tolist())
        plt.plot(x,y)
        plt.title(chrt_ttl + ' '+country_)
        plt.grid()
        plt.show()


    def work_with_countries(self, countries_: dict):
        dat = self.bp_DataFrame
        temp = dat[["Country", "Year", index_moc, clm_nm]]  # clm_nm == "oilprod_kbd"
        keys = list(main_oil_countries.keys())
        values = list(main_oil_countries.values())
        plt.grid()
        plt.title(chrt_ttl + ': ' + ', '.join(keys))
        for i in range(len(countries_)):
            C_Series = temp[temp[index_moc].str.contains(values[i])]
            print(len(C_Series))
            plt.plot(C_Series["Year"], C_Series[clm_nm], label= keys[i])
            # print(type(C_Series))
        plt.legend()
        plt.show()

    def work_with_main_countries(self):
        # дополнительно к work_with_countries идет выделение пог годам
        plt.subplots(figsize=(10, 5.3))
        main_oil_countries2 = create_main_oil_countries2()
        dat = self.bp_DataFrame
        print(type(dat))
        temp = dat[["Country", "Year", index_moc, clm_nm]]  # clm_nm == "oilprod_kbd"
        keys = []; values = []
        for i in range(main_oil_countries2.size):
            keys += [main_oil_countries2[i].full_name]
            values += [main_oil_countries2[i].shrt_name]
        print(keys)
        plt.title(chrt_ttl + ': ' + ', '.join(keys))
        plt.grid()

        temp2 = extract_in_years(temp, main_oil_countries2)

        for i in range(len(main_oil_countries2)):
            C_Series = temp2[temp2[index_moc].str.contains(values[i])]
            plt.plot(C_Series["Year"], C_Series[clm_nm], label=keys[i])
        plt.legend()
        plt.show()

