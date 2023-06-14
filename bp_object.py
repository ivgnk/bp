import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class bp_dat():
    bp_DataFrame: pd.DataFrame # полный набор данных
    curr_Series: pd.Series     # текущая серия для работы

    def __init__(self):
        # ввод из файла в папке dat
        bp_xlsx = pd.ExcelFile("dat/bp-stats-review-2022-consolidated-dataset-panel-format.xlsx")
        print(bp_xlsx.sheet_names)
        self.bp_DataFrame = bp_xlsx.parse()
        print('end __init__')

    def the_input(self):
        pass

    def the_work(self):
        pass

    def the_visualisation(self):
        pass

    def the_end(self):
        print('Нормальное завершение')

    # Частные методы
    def work_with_Saudi_Arabia(self):
        print('work_with_Saudi_Arabia')
        # SA_Series = self.bp_DataFrame["oilprod_kbd"]
        dat = self.bp_DataFrame
        temp = dat[["Country", "Year", "oilprod_kbd"]]
        SA_Series=temp[temp["Country"].str.contains("Saudi Arabia")]
        plt.plot(SA_Series["Year"], SA_Series["oilprod_kbd"])
        plt.grid()
        plt.show()

