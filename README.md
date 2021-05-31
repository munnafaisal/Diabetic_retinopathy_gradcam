# Diabetic_retinopathy_gradcam

![Test Image](https://github.com/munnafaisal/Diabetic_retinopathy_gradcam/blob/555a9d1ddaf269d82a1bd26a1c51fa610b2ca841/test_images/Screenshot%20from%202021-05-30%2020-14-47.png)

## A Short Description Of Project

This repo contains Gradcam visualization of Retinal fundus Images from a trained efficientnet_b5
model. Gradcam has been calculated from an intermediate layer which can highlight
Fat deposits, Isolated medium sized haemorrhages effectively. For more better visualization
Accumulated gradcams calculated from several layers can be very useful 

## Environment Setup:


### Installation instructions

_Run the commands in a terminal or command-prompt.

- Install `Python 3.6 or >3.6` for your operating system, if it does not already exist.

 - For [Mac](https://www.python.org/ftp/python/3.6.8/python-3.6.8-macosx10.9.pkg)

 - For [Windows](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe)

 - For Ubuntu/Debian

 ```bash
 sudo apt-get install python3.6
 ```

 Check if the correct version of Python (3.6) is installed.

 ```bash
 python --version
 ```

**Make sure your terminal is at the root of the project i.e. where 'README.md' is located.**

* Get `virtualenv`.

 ```bash
 pip install virtualenv
 ```

* Create a virtual environment named `.env` using python `3.6` and activate the environment.

 ```bash
 # command for gnu/linux systems
 virtualenv -p $(which python3.6) .env

 source .env/bin/activate
 ```
 
* Install python dependencies from requirements.txt.
 ```bash
  pip install -r requirements.txt
  ```

### How to Run 
 
Run the script from terminal using following command 

```bash
 python3 Test_gradcam.py 
 ```
Now in the project root directory you will find all the  Gauss,
Gradcam and Overlapped (Gauss+Gradcam) images.
	

## Contacts

1. Md. Faisal Ahmed Siddiqi (ahmedfaisal.fa21@gmail.com)
