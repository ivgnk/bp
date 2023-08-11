# https://het.as.utexas.edu/HET/Software/Astropy-0.4.2/

import astropy
from random import *

from astropy.table import Table
from astropy.io import fits
from astropy.io import ascii
from astropy.utils.data import download_file

from astropy import units as u
from astropy.units import imperial
from astropy.table import *

from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from astropy.modeling import Fittable1DModel, Parameter
from astroquery.sdss import SDSS
from astropy.convolution import *
#convolve, convolve_fft, Gaussian1DKernel, Box1DKernel, RickerWavelet1DKernel)
from astropy.modeling.models import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pnumpyscipy import *

import os

def work_with_demo_table():
    '''
    Взято из
    https://web.iucaa.in/~ace/attachments/programs/ADAP-2021/slides/lect8_slides.pdf
    https://web.iucaa.in/~ace/attachments/programs/ADAP-2021/slides/lect9_slides.pdf
    '''
    demo_table = Table.read("demo.txt", format = "ascii")
    demo_table.write('demo.fits', format='fits', overwrite=True)

    # print (demo_table)
    # you can supply options such as
    # max_lines, max_width, show_unit, show_name
    demo_table.pprint(show_name=False)
    # demo_table.show_in_browser()
    print (len(demo_table)) # Number of rows.
    print (demo_table.colnames) # The names of the columns.
    # print (demo_table.meta)
    # print (demo_table["name", "mag_b"]) # more than one column
    print ('----\n\n',demo_table)
    grouped_table = demo_table.group_by("name")

    print ('----\n\n')
    for i in range (len(grouped_table.groups)):
        print('\ngroup = ',i)
        print(grouped_table.groups[i]) # first group
    # grouped_table.groups.aggregate(np.mean)

# https://learn.astropy.org/tutorials/FITS-images.html

def work_with_image_fit_file(view:list):
    '''
    Взято из
    https://learn.astropy.org/tutorials/FITS-images.html
    '''
    if view[0]:
        # Opening FITS files and loading the image data
        image_file = "http://data.astropy.org/tutorials/FITS-images/HorseHead.fits"
        # hdulist = fits.open(image_file) # example.fits is a sample image on my comp

        hdu_list = fits.open(image_file)
        hdu_list.info()
        image_data = hdu_list[0].data
        print(type(image_data))
        print(image_data.shape)
        image_data = fits.getdata(image_file)
        print(image_data)

        # Viewing the image data and getting basic statistics
        plt.imshow(image_data, cmap='gray')
        cbar = plt.colorbar()
        plt.show()

        print('Min:', np.min(image_data))
        print('Max:', np.max(image_data))
        print('Mean:', np.mean(image_data))
        print('Stdev:', np.std(image_data))

        # Plotting a histogram
        print(type(image_data.flatten()))
        print(image_data.flatten().shape)
        plt.hist(image_data.flatten(), bins='auto')
        plt.show()

        # Displaying the image with a logarithmic scale
        plt.imshow(image_data, cmap='gray', norm=LogNorm())
        cbar = plt.colorbar(ticks=[5.e3,1.e4,2.e4])
        cbar.ax.set_yticklabels(['5,000','10,000','20,000'])
        plt.show()

    if view[1]:
        # Basic image math: image stacking
        base_url = 'http://data.astropy.org/tutorials/FITS-images/M13_blue_{0:04d}.fits'
        image_list = [download_file(base_url.format(n), cache=True)
                      for n in range(1, 5+1)]
        image_concat = [fits.getdata(image) for image in image_list]
        print(type(image_concat))
        final_image = np.sum(image_concat, axis=0)

        # Writing image data to a FITS file
        outfile = 'stacked_M13_blue.fits'
        hdu = fits.PrimaryHDU(final_image)
        print(type(hdu.header))
        hdu.writeto(outfile, overwrite=True)
        llst = list(hdu.header)
        for s in llst:
            print(s)
        curr_dir = os.getcwd(); print(curr_dir)
        curr_dat = "\\".join([curr_dir, outfile])
        print(curr_dat)
        dlf = fits.open(curr_dat)
        dlf.info()
        # print(dlf.h)

def work_with_user_defined_model():
    '''
    Взято из
    https://learn.astropy.org/tutorials/User-Defined-Model.html
    '''
    # Fit an Emission Line with a Gaussian Model

    spectrum = SDSS.get_spectra(plate=1349, fiberID=216, mjd=52797)[0]
    print('type(spectrum) = ',type(spectrum))
    units_flux = spectrum[0].header['bunit']
    print('type(units_flux) = ',type(units_flux))
    flux = spectrum[1].data['flux']
    print('type(flux) = ',type(flux))
    lam = 10 ** (spectrum[1].data['loglam'])
    print('type(lam) = ',type(lam))

    gausian_model = models.Gaussian1D(1, 6563, 10) + models.Polynomial1D(degree=1)
    fitter = fitting.LevMarLSQFitter()
    gaussian_fit = fitter(gausian_model, lam, flux)

    plt.figure(figsize=(8,5))
    plt.plot(lam, flux, color='k')
    plt.plot(lam, gaussian_fit(lam), color='darkorange')
    plt.xlim(6300,6700)
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux ({})'.format(units_flux))
    plt.show()

def read_text_file():
    # Read in catalog information from a text file and plot some parameters
    # https://learn.astropy.org/tutorials/plot-catalog.html

    tbl = ascii.read("simple_table.csv")
    print(tbl)
    print(tbl["ra"])


def test_smooth_oil_price():
    tbl = read_oil_price("oil_price_1861-2021.txt")
    # smooth_oil_price_triang(tbl, llen=5)
    # smooth_oil_price_gauss(tbl, lst_stddev = [1,3,5])

    # n = 10**(-20);  param = [(i*1.3+1)*n for i in range(5)]
    # print(param)
    # smooth_oil_price_curr1DKernel(tbl, param)

    llst = create_kernel_list()
    i = 0
    for curr_ker in llst:
        print(i)
        smooth_oil_price_w_kernel5(tbl, curr_ker)
        i = i+1

def read_oil_price(fname:str):
    print(inspect.currentframe().f_code.co_name)
    tbl = ascii.read(fname)
    # print(tbl)
    # print(tbl["Year"])
    # print(tbl[oil_pr_sht_clm_nm[0]])
    print(type(tbl))
    print(len(tbl))
    return tbl

def my_triang_kernel(param):
    llen = SMA_w[param]
    x = calc_triang_weights_for_1Dfilter(llen)
    weights = norm_weights_for_1Dfilter(x)
    return weights

def create_kernel_list()->list:
    llst=[]
    llst.append(the_kernel(my_triang_kernel,[0, 1, 2, 3, 4],'my_triang'))
    llst.append(the_kernel(Gaussian1DKernel,[1, 2, 3, 4, 5],'Gaussian1DKernel'))
    llst.append(the_kernel(Box1DKernel, SMA_w ,'Box1DKernel'))

    return llst

def test_kernel():
    llst = [1,1.1,1.20,1.30,1.40]
    for i in range(5):
        weights = RickerWavelet1DKernel(llst[i])
        print(weights.array)



def smooth_oil_price_w_kernel5(tbl, thekernel:the_kernel):
    print(thekernel.kernel_nms)
    x = tbl["Year"]
    pr_curr = tbl[oil_pr_sht_clm_nm[1]]
    pr_2021 = tbl[oil_pr_sht_clm_nm[2]]
    plt.subplots( nrows = 1, ncols =2, figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.plot(x, pr_curr, label='curr ini', linewidth=5)
    plt.title('Price curr, USD, Kernel='+thekernel.kernel_nms)
    plt.subplot(1, 2, 2)
    plt.plot(x, pr_2021, label='2021 ini', linewidth=5)
    plt.title('Price $2021, USD, Kernel='+thekernel.kernel_nms)
    for i in range(5):
        kernel_name = thekernel.kernel_nm
        kernel_param = thekernel.kernel_prm
        weights = kernel_name(kernel_param[i])
        res_curr= convolve(pr_curr, weights, boundary='extend')
        res_2021= convolve(pr_2021, weights, boundary='extend')
        if type(weights) == np.ndarray:
            # print(i, ' ', weights, ' ')
            w = str(weights.size)
        else: # настоящий kernel
            # print(i, ' ', weights.array, ' ')
            arr = weights.array
            w = str(arr.size)

        plt.subplot(1, 2, 1)
        plt.plot(x, res_curr, label='curr '+w)
        plt.subplot(1, 2, 2)
        plt.plot(x, res_2021, label='2021 '+w)
    plt.subplot(1, 2, 1);     plt.grid();     plt.legend()
    plt.subplot(1, 2, 2);     plt.grid();     plt.legend()
    plt.show()

def smooth_oil_price_gauss(tbl, lst_stddev:list):
    '''
    свертка с использованием Gaussian1DKernel
    https://docs.astropy.org/en/stable/convolution/index.html
    '''
    print(inspect.currentframe().f_code.co_name)
    print(type(Gaussian1DKernel))
    x = tbl["Year"]
    pr_curr = tbl[oil_pr_sht_clm_nm[1]]
    pr_2021 = tbl[oil_pr_sht_clm_nm[2]]
    plt.subplots( nrows = 1, ncols =2, figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.plot(x, pr_curr, label='curr ini', linewidth=5)
    plt.title('Price curr, USD')
    plt.subplot(1, 2, 2)
    plt.plot(x, pr_2021, label='2021 ini', linewidth=5)
    plt.title('Price $2021, USD')
    for i in range(len(lst_stddev)):
        gauss = Gaussian1DKernel(stddev=lst_stddev[i])
        res_curr= convolve(pr_curr, gauss, boundary='extend')
        res_2021= convolve(pr_2021, gauss, boundary='extend')
        w = 'std = ' + str(lst_stddev[i])
        plt.subplot(1, 2, 1)
        plt.plot(x, res_curr, label='curr '+w)
        plt.subplot(1, 2, 2)
        plt.plot(x, res_2021, label='2021 '+w)
    plt.subplot(1, 2, 1);     plt.grid();     plt.legend()
    plt.subplot(1, 2, 2);     plt.grid();     plt.legend()
    plt.show()

def smooth_oil_price_triang(tbl, llen:int):
    llst:list = calc_lst_of_triang_weights(llen, win_size_lst=SMA_w) # SMA_w
    x = tbl["Year"]
    pr_curr = tbl[oil_pr_sht_clm_nm[1]]
    pr_2021 = tbl[oil_pr_sht_clm_nm[2]]
    plt.subplots( nrows = 1, ncols =2, figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.plot(x, pr_curr, label='curr ini', linewidth=5)
    plt.title('Price curr, USD')
    plt.subplot(1, 2, 2)
    plt.plot(x, pr_2021, label='2021 ini', linewidth=5)
    plt.title('Price $2021, USD')
    for i in range(llen):
        weights: np.ndarray = llst[i]
        # print(i,' ' , weights, ' ', np.sum(weights))
        res_curr= convolve(pr_curr, weights, boundary='extend')
        res_2021= convolve(pr_2021, weights, boundary='extend')
        w = str(weights.size)
        plt.subplot(1, 2, 1)
        plt.plot(x, res_curr, label='curr '+w)
        plt.subplot(1, 2, 2)
        plt.plot(x, res_2021, label='2021 '+w)
    plt.subplot(1, 2, 1);     plt.grid();     plt.legend()
    plt.subplot(1, 2, 2);     plt.grid();     plt.legend()
    plt.show()

def smooth_oil_price_curr1DKernel(tbl, param:list):
    '''
    свертка с использованием RickerWavelet1DKernel
    https://docs.astropy.org/en/stable/convolution/index.html
    https://docs.astropy.org/en/stable/api/astropy.convolution.RickerWavelet1DKernel.html
    '''
    print(inspect.currentframe().f_code.co_name)
    x = tbl["Year"]
    pr_curr = tbl[oil_pr_sht_clm_nm[1]]
    pr_2021 = tbl[oil_pr_sht_clm_nm[2]]
    plt.subplots( nrows = 1, ncols =2, figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.plot(x, pr_curr, label='curr ini', linewidth=5)
    plt.title('Price curr, USD')
    plt.subplot(1, 2, 2)
    plt.plot(x, pr_2021, label='2021 ini', linewidth=5)
    plt.title('Price $2021, USD')

    for i in range(len(param)):
        gauss = RickerWavelet1DKernel(param[i])
        res_curr= convolve(pr_curr, gauss, boundary='extend')
        res_2021= convolve(pr_2021, gauss, boundary='extend')
        w = 'std = ' + str(param[i])
        plt.subplot(1, 2, 1)
        plt.plot(x, res_curr, label='curr '+w)
        plt.subplot(1, 2, 2)
        plt.plot(x, res_2021, label='2021 '+w)
    plt.subplot(1, 2, 1);     plt.grid();     plt.legend()
    plt.subplot(1, 2, 2);     plt.grid();     plt.legend()
    plt.show()

def create_Gaussian1D_kernel(x_sz:int, ampl_:float=1, mn_:float=0, std_:float=2,is_view:bool=False)->np.ndarray:
    '''
    x_sz - длина результирующего numpy.ndarray
    ampl - амплитуда, mn - среднее 0, std - стандартное отклонение
    На основе Model1DKernel
    https://docs.astropy.org/en/stable/api/astropy.convolution.Model1DKernel.html#astropy.convolution.Model1DKernel
    '''
    gauss = Gaussian1D(amplitude=ampl_, mean=mn_, stddev=std_)
    gauss_kernel = Model1DKernel(gauss, x_size=x_sz)
    if is_view:
        print(gauss)
        print('\n')
        print(gauss.amplitude)
        print(gauss.mean)
        print(gauss.stddev)
        print('\n')
        print(gauss_kernel.array,'  ',type(gauss_kernel.array))
    return gauss_kernel.array

def test_create_Gaussian1D_kernel():
    pass

def test_astropy_units():
    '''
    https://docs.astropy.org/en/stable/units/index.html#module-astropy.units
    https://docs.astropy.org/en/stable/units/index.html#getting-started
    https://docs.astropy.org/en/stable/units/index.html#using-astropy-units

    https://kbarbary-astropy.readthedocs.io/en/latest/units/composing_and_defining.html !!!
    '''
    print('\n--- Compare ---')
    tm = 10*u.meter
    print(type(u.meter), ' ', u.meter,' ',tm)
    tm2 = 10*u.m
    print(type(u.m), ' ', u.m,' ',tm2)
    tm3 = 10*u.mm
    print(type(u.mm), ' ', u.mm,' ',tm3)
    t = 10*u.year
    print(type(u.year), ' ', u.year,' ',t)
    ed = u.meter/u.second; t = 10*ed
    print(type(ed), ' ', ed,' ',t)
    ts = 10*u.s
    print(type(u.s), ' ', u.s,' ',t)
    t2 = 10*u.second
    print(ts,' and  ',t2,' = ',ts == t2)
    print(ts,' and  ',tm,' = ',ts == tm)
    print(tm2,' and  ',tm,' = ',tm2 == tm)
    print(tm3,' and  ',tm,' = ',tm3 == tm)
    tm4 = 1000*tm3
    print(tm4,' and  ',tm,' = ',tm4 == tm)
    t = 10*imperial.mile
    # print(type(u.barrel), ' ', u.barrel,' ',t)

    print('\n--- Calculation ---')
    print(tm2)
    b = tm2*10
    print(b,'  ', b.value,' ',b.unit)


    print('\n--- New Units ---')
    bakers_fortnight = u.def_unit('bakers_fortnight', 13 * u.day)
    a = 10*bakers_fortnight
    print(a,'\n  value = ', a.value,', unit = ',a.unit)

    barrel = u.def_unit('barrel', 42 * imperial.gallon)
    print(barrel)
    a = 20*barrel
    print(a,'\n  value = ', a.value,', unit = ',a.unit)

def test_astropy_tables():
    '''
    https://docs.astropy.org/en/stable/table/
    '''
    # --------- VAR 1
    print('\n --------- VAR 1')
    a = np.array([1, 4, 5], dtype=np.int32)
    b = [2.0, 5.0, 8.5]
    c = ['x', 'y', 'z']
    d = [10, 20, 30]

    tbl = Table([a, b, c, d],
               names=('a', 'b', 'c', 'd'),
               meta={'name': 'first table'})
    print(tbl)

    # --------- VAR 2
    print('\n --------- VAR 2')
    n = 10
    a = np.array([i for i in range(n)])
    b = np.array([random() for i in range(n)])
    c = 10*b

    tbl = Table([a, b, c],
               names=('a', 'b', 'c'),
               meta={'name': 'first data'})
    print(tbl)
    print(len(tbl))
    a = tbl["a"]
    print('\n a = ',a)
    b = a[3]
    print('\n b = ',b)
    b = tbl[3]
    print('\n b = ',b)
    a = tbl["a"][3]
    print('\n a = ',a)
    a = tbl[0:3]
    print('\n a = ',a)


    a = tbl["a"][0]
    b = tbl["b"][0]
    c = tbl["c"][0]
    print('\n a = ', a,
          '\n b = ', b,
          '\n c = ', c)

    # print('\n a.array = ', a.array)



if __name__ == "__main__":
    # work_with_demo_table()
    # work_with_image_fit_file([False, True])
    #work_with_user_defined_model()
    # read_text_file()

    # test_smooth_oil_price()
    # create_Gaussian1D_kernel(9, True)

    # test_astropy_units()
    test_astropy_tables()
    #view_astropy_units()