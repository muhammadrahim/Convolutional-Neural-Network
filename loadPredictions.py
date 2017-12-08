from numpy import genfromtxt
from numpy.distutils.conv_template import file
import os
#os.chdir('')
## Edit the path in the loading method below so it points to where you saved the data or use change dir (os.chdir above to switch to your working directory
# Ive provided a few predictions to work with. 2008_006213 is the file name if you want to compare to the original text file. This can be changed to load other predictions.
y_predicted = genfromtxt('./good predictions/y_pred_2008_006213.csv', delimiter=',')


