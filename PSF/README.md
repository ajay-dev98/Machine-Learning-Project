# Project Title


## How to Reproduce the Runs

Step 1 : Download all these files to a common folder.

Step 2 : Run the ellipse_angle.py file. This will create a text file which contains the dataset for the regression. Currently the function is set as cos. If you wish to create dataset for a different function then change accordingly.

Step 3 : Open cnn_simple_regression_ellipse.py and in line 25 put the name of the text file that was created in last step. By default its been set correctly if u did not change the file name earlier. Then run the file. It will show the mean error and standard deviation. A text file will also be saved which needs to be used in the next step.

Step 4 : Open the simulate_star.py file and run it. If you changed the name of the textfile created in last step then edit the name accordingly. Running this file will create images for predicted and actual psf  in a folder named images.

Step 5 : Open the compare_images file and run it. It will calculate the MSE between the PSF with predicted and actual angle and print it. 




## Authors

* **Ajay Dev** - *ajay.dev@niser.ac.in* 
* **Ashish Mishra** - *ashish.mishra@niser.ac.in* 

Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
