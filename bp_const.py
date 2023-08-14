'''
Константы для программы
'''
from copy import deepcopy
from dataclasses import dataclass
from typing import *
import numpy as np
import math

from astropy.table import *


dat_dir = 'dat'
res_dir = 'res'
panel_xls_fname = 'bp-stats-review-2022-consolidated-dataset-panel-format.xlsx'
my_panel_xls_fname = 'my_bp-stats-review-2022-consolidated-dataset-panel-format.xlsx'
main_xls_fname = 'bp-stats-review-2022-all-data.xlsx'
main_xls_oil_price_sheet = 'Oil crude prices since 1861'


opec_list = ['Algeria', 'Angola', 'Equatorial Guinea', 'Gabon',
                 'Iran', 'Iraq', 'Kuwait', 'Libya',
                 'Nigeria', 'Republic of Congo','Saudi Arabia', 'United Arab Emirates','Venezuela']

opecplus_list = ['Mexico', 'Azerbaijan', 'Kazakhstan', 'Russian Federation',
                 'Oman', 'Bahrain', 'Brunei', 'Malaysia',
                 'South Sudan', 'Sudan']

Col_Opec_plus: str = 'OPEC+'

## Для self.panel_frame_restore_region
spec_region_list = ['World', 'Africa', 'OPEC',
                    'OECD', 'Non-OPEC', 'Non-OECD', 'Middle East',
                    'Europe', 'CIS', 'Asia Pacific',
                    'North America','S. & Cent. America']
spec_region_list2 = ['Other Caribbean','Other South America']

## Для self.self.the_oil_price_the_visualisation
oil_pr_sht_clm_nm = ['Year', '$ money of the day', '$ 2021']
theYear = 'Year'
themotd = '$ money of the day'
thed2021 = '$ 2021'

## Для self.the_oil_production_in_year_visualisation
clm_nm = 'oilprod_kbd'
chrt_ttl = 'Добыча нефти всех видов, kb/d ('+clm_nm+')'
years_prod_visu = [1965, 2021]
country_prod_visu = ['Total North America','Total S. & Cent. America','Total Europe','Total CIS',
                     'Total Middle East', 'Total Africa', 'Total Asia Pacific']

## Для self.work_with_countries
main_oil_countries: dict = {'Saudi Arabia':'SAU', 'US': 'USA'}
index_moc: str = 'ISO3166_alpha3'

## Для self.work_with_countries2

oil_price_years:list = [i for i in range(1861, 2022, 1)] # годы известных цен
oil_gas_prod_years:list = [i for i in range(1965, 2022, 1)] # годы добычи
oil_gas_rsrv_yearsnewdat = [i for i in range(1980, 2022, 1)] # годы запасов

SAU_oilprod_kbd = [ 2219,  2618, 2825,  3081, 3262,   3851,  4821,  6070,  7693,  8618,  7216,  8762,  9419,  8554, 9842,
                   10270, 10256, 6961,  4951, 4534,   3601,  5208,  4450,  5656,  5636,  7106,  8820,  9092,  8893,
                    8983,  8974, 9087,  9005, 9267,   8524,  9121,  8935,  8207,  9628, 10306, 10839, 10671, 10269,
                   10665,  9709, 9865, 11079, 11622, 11393, 11519, 11998, 12406, 11892, 12261, 11832, 11039, 10954]
oil_price_usd_curr = [ 0.49, 1.05, 3.15, 8.06, 6.59, 3.74, 2.41, 3.63 , 3.64, 3.86, 4.34, 3.64, 1.83, 1.17, 1.35, 2.56,
                     2.42, 1.19, 0.86, 0.95, 0.86, 0.78, 1.00, 0.84, 0.88, 0.71, 0.67, 0.88, 0.94, 0.87, 0.67, 0.56,
                     0.64, 0.84, 1.36, 1.18, 0.79, 0.91, 1.29, 1.19, 0.96, 0.80, 0.94, 0.86, 0.62, 0.73, 0.72, 0.72,
                     0.70, 0.61, 0.61, 0.74, 0.95, 0.81, 0.64, 1.10, 1.56, 1.98, 2.01, 3.07, 1.73, 1.61, 1.34, 1.43,
                     1.68, 1.88, 1.30, 1.17, 1.27, 1.19, 0.65, 0.87, 0.67, 1.00, 0.97, 1.09, 1.18, 1.13, 1.02, 1.02,
                     1.14, 1.19, 1.20, 1.21, 1.05, 1.12, 1.90, 1.99, 1.78, 1.71, 1.71, 1.71, 1.93, 1.93, 1.93, 1.93,
                     1.90, 2.08, 2.08, 1.90, 1.80, 1.80, 1.80, 1.80, 1.80, 1.80, 1.80, 1.80, 1.80, 1.80, 2.24, 2.48,
                     3.29, 11.58, 11.53, 12.80, 13.92, 14.02, 31.61, 36.83, 35.93, 32.97, 29.55, 28.78, 27.56, 14.43,
                    18.44, 14.92, 18.23, 23.73, 20.00, 19.32, 16.97, 15.82, 17.02, 20.67, 19.09, 12.72, 17.97, 28.50,
                    24.44, 25.02, 28.83, 38.27, 54.52, 65.14, 72.39, 97.26, 61.67, 79.50, 111.26, 111.67, 108.66, 98.95,
                    52.39, 43.73, 54.19, 71.31, 64.21, 41.84, 70.91]

oil_price_usd_2021 = [14.06, 27.11, 65.94, 132.83, 110.97, 65.84, 44.45, 70.29, 70.49, 78.68, 93.38, 78.32, 39.37, 26.65,
                      31.69, 61.97, 58.58, 31.78, 23.79, 25.37, 22.97, 20.83, 27.66, 24.10, 25.25, 20.37, 19.22, 25.25,
                      26.97, 24.96, 19.22, 16.07, 18.36, 25.03, 42.14, 36.56, 24.48, 28.19, 39.97, 36.87, 29.74, 23.83,
                      26.97, 24.67, 17.79, 20.94, 19.92, 20.66, 20.08, 16.87, 16.87, 19.77, 24.78, 20.84, 16.30, 26.06,
                      31.46, 34.00, 30.06, 39.64, 25.01, 24.84, 20.31, 21.63, 24.79, 27.48, 19.36, 17.66, 19.17, 18.44,
                      11.04, 16.48, 13.37, 19.32, 18.28, 20.35, 21.26, 20.74, 18.99, 18.81, 20.03, 18.89, 17.95, 17.79,
                      15.09, 14.83, 22.00, 21.38, 19.32, 18.37, 17.02, 16.66, 18.66, 18.57, 18.64, 18.37, 17.45, 18.61,
                      18.45, 16.59, 15.56, 15.39, 15.21, 15.00, 14.75, 14.35, 13.94, 13.38, 12.70, 11.99, 14.30, 15.35,
                      19.17, 60.81, 55.47, 58.21, 59.41, 55.65, 112.69, 115.68, 102.30, 88.42, 76.79, 71.69, 66.29, 34.08,
                      42.00, 32.65, 38.04, 46.98, 38.01, 35.64, 30.40, 27.62, 28.90, 34.09, 30.79, 20.19, 27.92, 42.83,
                      35.72, 36.00, 40.55, 52.43, 72.25, 83.63, 90.36, 116.91, 74.40, 94.35, 128.01, 125.88, 120.72,
                      108.17, 57.20, 47.16, 57.22, 73.50, 65.00, 43.80, 70.91]

# Для фильтраций временных рядов добычи нефти, баррели
SMA_all_w = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
SMA_w: list = [3, 7, 13, 25, 51]  # SMA - Simple Moving  Average,  размер окон


# https://habr.com/ru/articles/415829/
@dataclass
class CountryYears:
    full_name:str
    shrt_name:str
    begYear:int
    endYear:int
    s_w_s:list  #SMA_window_size , SMA - Simple Moving  Average

# https://habr.com/ru/articles/415829/
@dataclass
class the_kernel:
    kernel_nm:Any
    kernel_prm:list
    kernel_nms:str


@dataclass
class Line_:
    x:np.ndarray
    y:np.ndarray
    name:str  # название линии (фильтра)
    param:str # пароаметры линии (фильтра)
    num:int   # номер линии (фильтра)
    currnum:int # номер подварианта линии (фильтра)

def create_main_oil_countries2()->np.ndarray:
     def_SMA_window_size:list = [3, 7, 13, 25]
     main_oil_countries2 = np.arange(3, dtype=CountryYears)
     main_oil_countries2[0] = CountryYears(full_name = 'Saudi Arabia', shrt_name='SAU', begYear=1970, endYear=2020,
                                           s_w_s = def_SMA_window_size)
     main_oil_countries2[1] = CountryYears(full_name = 'US', shrt_name='USA', begYear=1965, endYear=2021,
                                           s_w_s = def_SMA_window_size)
     main_oil_countries2[2] = CountryYears(full_name = 'Russian Federation', shrt_name='RUS', begYear=1991, endYear=2021,
                                           s_w_s = def_SMA_window_size)
     return main_oil_countries2

def calc_proc_growth(llst:list)->list:
    growth_ = deepcopy(llst)
    llen = len(oil_price_years)
    for i in range(llen):
        if i==0:
            growth_[i] =math.nan
        else:
            growth_[i] = (llst[i]-llst[i-1])/llst[i-1]*100
    return growth_

def calc_sum_table_by_clmn(tbl:Table, tbl_clm_nm:str):
    llen = len(tbl)
    ssum = 0
    for i in range(llen):
        ssum += tbl[tbl_clm_nm][i]
    return ssum

def calc_oil_price_stat():
    grw_curr = calc_proc_growth(oil_price_usd_curr)
    grw_2021 = calc_proc_growth(oil_price_usd_2021)

    tbl_curr = Table([oil_price_years, oil_price_usd_curr, grw_curr],
               names=('oil_price_years', 'oil_price_usd_curr', 'grw_curr, %'),
                      meta={'name': 'oil_price_usd_curr growth'})
    tbl_curr['grw_curr, %'].info.format = '8.2f'
    print('\n Table name =',  tbl_curr.meta['name'])
    print(tbl_curr)

    tbl_2021 = Table([oil_price_years, oil_price_usd_2021, grw_2021],
               names=('oil_price_years', 'oil_price_usd_2021', 'grw_2021, %'),
                      meta={'name': 'oil_price_usd_2021 growth'})
    tbl_2021['grw_2021, %'].info.format = '8.2f'
    print('\n Table name =',  tbl_2021.meta['name'])
    print(tbl_2021)

    print(tbl_2021['oil_price_usd_2021'].groups.aggregate(np.mean))
    print(tbl_2021['oil_price_usd_2021'].groups.aggregate(np.sum))
    print(calc_sum_table_by_clmn(tbl_2021,'oil_price_usd_2021'))

    # calc_sum_by_table(tbl:Table, tbl_clm_nm:str):


def test_data():
    print('len(oil_price_years)    = ', len(oil_price_years))
    print('len(oil_price_usd_curr) = ', len(oil_price_usd_curr))
    print('len(oil_price_usd_2021) = ', len(oil_price_usd_2021))
    calc_oil_price_stat()

if __name__ == "__main__":
    test_data()