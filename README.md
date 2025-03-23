# Asymptotic-CBs-for-CPRF
This repositary contains simulations of asymptotic confidence bands for centered purely random forests used in my PhD thesis.
## Structure
- `src/forest.py` contains the source code for the random forests and regression trees
- `src/functions.py` contains several custom functions
## Usage
Simulations of asymptotic confidence bands are done in `Asymp CBs Setting p=2 v02 w Histo.ipynb` and `Asymp CBs Setting p=4 v01.ipynb` for two different regresison models. The bootstrap simulations are done in `Bootstrap CBs Setting 1 v01 inner parallel.ipynb`. The code for the plots is in `Plots.ipynb`. The results can be found in `Simulation results.txt` and `Bootstrap results.txt`. `py versions.txt` contains the versions of python packages used. 

## Use of GAI
In creating this repository, generative AI was used to
- Initially create an object-oriented programming structure for generic regression trees.
- Researching the appropriate Python packages and methods for specific tasks, and explaining their use.
