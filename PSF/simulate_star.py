# ajay.dev
# date: 14-11-20
import sys
import os
import numpy as np
from astropy.table import Table
from photutils.datasets import (make_random_gaussians_table, make_noise_image, make_gaussian_sources_image,apply_poisson_noise)
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import pandas as pd
import datasets_ellipse
import models_ellipse

# Take necessary parameters into account
D_o =  100.0 # Outer Diameter of the telescope in cm.
D_i =  0.00   # Inner Diameter of the telescope in cm.
Ar = 0.25*np.pi*(D_o**2 - D_i**2)
B = 1500.0 # bandwidth of the Guider camera band. 1000 A is the approximate bandwidth of the V band filter
e = 0.1 # efficiency of the telescope plus guider system. Reasonable assumption.
f_v= 1000.0 # Flux of 0 Mag V band star. photons/cm2/t/Angstrom. # need to check this no. Checked, it is correct.
pixel_size = 0.5 # in arcseconds.
seeing=1.5
m_star=16.0
exp_time=1200
relz= 1 #str(sys.argv[1])

# Magnitude of the source
def find_stats(m_star,t):
    n_p = f_v*(10**(-m_star/2.5))*Ar*B*t*e
    N = int(n_p) # No. of photons received from a source.
    return N


# mention the parent directory
parent_dir="/home/ashish/MACHINE LEARNING/ash"
# construct the path to the input .txt file that contains information
# and then load the dataset
print("[INFO] loading PSF...")
#joining the pathe with a directory named images
path = os.path.join(parent_dir, "images")
# if images named directory doesn't exist then create one or do nothing.
if not os.path.exists('images'):
    os.makedirs(path)
# Read data from the predicted file 
df = pd.read_csv("predicted_cos.txt")
# Store actual and predicted values in different arrays.
actual = df.to_numpy()[:,0]
pred = df.to_numpy()[:,1]

# All measurements are in Pixels
f= find_stats(m_star,exp_time)
sigma_psf = seeing/(pixel_size*2.355)
sources = Table()
sources['flux'] = [f]
sources['x_mean'] = [15]
sources['y_mean'] = [15]
sources['x_stddev'] = sigma_psf*np.ones(1)
sources['y_stddev'] = sigma_psf*np.ones(1)*1.5
sources['id'] = [1]
tshape = (31, 31)

# define source angle as the actual value
# and crete a gaussian source image
# the save the psf image obtained in the 'path'
for i in range(len(actual)):
    sources['theta'] = actual[i]
    image1 = make_gaussian_sources_image(tshape, sources)
    image_tmp = apply_poisson_noise(image1) 
   # plt.figure(1)
    plt.imshow(image_tmp, origin='lower')
    plt.imsave(os.path.join(path, str("actual_{}.png".format(i))) ,image_tmp)
   # plt.show()
    
# Similar to previous step but creating psf of the predicted data. 
for i in range(len(pred)):
    sources['theta'] = pred[i]
    image1 = make_gaussian_sources_image(tshape, sources)
    image_tmp = apply_poisson_noise(image1) 
   # plt.figure(1)
    plt.imshow(image_tmp, origin='lower')
    plt.imsave(os.path.join(path, str("pred_{}.png".format(i))) ,image_tmp)
   # plt.show()



