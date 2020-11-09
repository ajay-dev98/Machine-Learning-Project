# import the necessary packages
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from skimage.draw import (ellipse)

# the range of angles in which ellipse should be created
x=np.arange(np.deg2rad(0),np.deg2rad(180),np.deg2rad(0.1))
y=x*180/np.pi

# labels for each image are saved in a file
with open('label.txt', 'w') as file:
    for data in y:
        file.write(str(data)+'\n')

#creating and saving ellipse images
for i in range(len(x)):
    img = np.zeros((500, 500, 3), dtype=np.double)
    rr, cc = ellipse(250, 250, 100, 200, rotation=x[i])
    img[rr, cc, :] = 1
    plt.show()
    plt.imsave('{}.png'.format(i),img,cmap=cm.gray)
