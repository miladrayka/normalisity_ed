# Investigating Normalisity of Error Distribution by *Normalisity_ED* Web App

If the distribution of residual errors for a benchmark dataset is zero-centered normal, statistics such as the mean absolute error (*MAE* ) or the root mean squared deviation (*RMSD* ) are sufficient for describing error distribution, but if it is not normal, more information is necessary to provide. 
This package (a Web Aapp) is devised for investigating normalisity property of error distribution by several statistical methods (Shapiro-Wilk, Skewness, Kurtosis, Q95, QQ-plot, Histogram-Normal plot, Residual Error Distribution plot.)

## Contact
Milad Rayka, Chemistry and Chemical Engineering Research Center of Iran, milad.rayka@yahoo.com

## Installation 

Below packages should be installed for using *Normalisity_ED*. Dependecies:
- python >= 3
- streamlit
- matplotlib
- seaborn
- numpy
- scipy
- pandas

For installing first make a virtual environment and activate it.

Using *conda* :

`conda create -n env`

Or (Windows):

       python py -m venv env
       .\env\Scripts\activate

Or (macOS and Linux):

       python3 -m venv env
       source env/bin/activate

Which *env* is the location to create the virtual environment. Now you can install packages:

`pip install -r requirements.txt`

## Usage

This *Normalisity_ED* is a Web App based on Streamlit package. For using *Normalisity_ED*, you just need run the following code:

`streamlit run normalisity.py`

## Reference

All texts are extracted from *Mach. Learn.: Sci. Technol., 2020, 1, 035011* paper.
