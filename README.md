# Finding Good Photo
( describing how to run the code and recreate the results that you present)
## 1. Dataset
Obtain the dataset (and code) from google drive: https://drive.google.com/file/d/1Vbjp-fV72_XiwB2sgdgBTmrJWa-L4cPN/view?usp=sharing
Unzip the data.zip and place the 'data/' folder alongside 'code/'.

## 2. Setup
1. UsingMiniconda,createacondaenvironmentusingtheappropriatecommand.OnWindows, open the installed "Conda prompt" to run this command. On MacOS or Linux, you can just use a terminal window to run the command:
conda env create -f environment.yml

2. This should create an environment named finalProject. Activate it using the following Windows command:
conda activate finalProject

or the following MacOS/Linux command:
conda activate finalProject

Use the following command to install the cvlib:
pip install cvlib

## 3. Run the code
Use ‘cd’ command go to code directory.
If you want to test on different data, feel free to modify the paths variable in test function of each part.

Test on Blur:
python blur.py

Test on exposureness:
python exposure.py

Test on orientation:
python orientation.py

Test on svm facial expression:
python expressionSVM.py

Test on neural network facial expression:
python facial.py

Test on object detection:
python object.py

Run the main program using command:
python NN.py

## 4. Results
The results will be displayed on the screen or saved in the directory after the program finish
