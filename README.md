# simulateLQRNetworks

For citing this code, the following DOI can be used:
[![DOI](https://img.shields.io/badge/DOI-Zenodo-darkblue)](https://doi.org/10.5281/zenodo.18163693)


This repository contains the code associated with the cell reports article "a model of neural population dynamics for flexible sensorimotor control" to simulate LQR-RNNs

The code that can run different tasks mentioned in the article (center out reaching, force fields, inertial loads, bar target (wider target task), shooting task)

### First step is to download the raw datafolder: 
To run the simulations or plot the data, download the folder named "datastore" from the following button 
[![Dataset](https://img.shields.io/badge/Dataset-Download%20datafolder-brightgreen)](https://doi.org/10.6084/m9.figshare.29149901) https://doi.org/10.6084/m9.figshare.29149901

Replace the dummy placeholder datafolder that exists in this repository with the above download


## Software Requirements
To get started install Anaconda python package and other dependencies such as: h5py, numpy, argpars, sklearn, pickle, json.

## To simulate a prepare to reach task to 8 targets
Then you can just run the python notebook file "reachingTask.ipynb" to simulate the task on a randomly connected RNN (dense RNN or an ISN network) whose weights are stored in the folder "datastore/WeightsData"

The simulation results are stored inside datastore/SimulationData/8dirreach_task/denseRNN/denseRNN_results_singlesimulation.hdf5

If you are using VScode, you can install a HDF5 viewer plugin so that you can visualize the contents of the hdf5 files in which we work in this code.

## To run the analysis:
Run the file named "analyze_plot_results_8dirtask.ipynb". This will plot the hand paths, kinematics, neural activities and also the PCA trajectories in the movement and preparation periods of the prepare-to-reach task.


## To simulate variations in reaching (dynamics and task objectives)
Run any of the respective python notebook files "forcefieldTask.ipynb", "inertialTask.ipynb", "widetargetTask.ipynb", "shootingTask.ipynb".

## To analyze and plot variations in reaching (dynamics and task objectives)
Run the python notebook "analyze_plot_results_taskvariations.ipynb".

Within this notebook in Cell 3 you can select the task that you want to analyze (forcefield, inertial, widetarget, shooting) and comment the remaining. Running the notebook should plot results similar to those in the article.

## Contact
For any queries please contact: hari.kalidindi at donders.ru.nl
