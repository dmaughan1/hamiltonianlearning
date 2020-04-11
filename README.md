# hamiltonianlearning

This repository contains a Numpy implementation of the Hamiltonian learning framework outlined in https://arxiv.org/abs/1911.12548

To be brief, the code maximizes the likelihood that the measured data was produced by the learning Hamiltonian via a descent method.

I hope to add a cleanly written Tensorflow implementation that can run on a GPU in the future. This is not a top priority for me at the moment. It would allow for testing on larger hamiltonian matrices.

The code requires NumPy, SciPy, and Python3. Save both files (hamiltonianlearning.py and hlFunctions.py) to the same folder and run the file hamiltonianlearning.py. This will automatically simulate the learning of a random Hamiltonian matrix. The code will run in its entirety in a few seconds. 

You can simulate find the hamiltonians of the hyperfine splitting of hydrogen, cyclobutadiene, or a random hamiltonian by setting the 'hamnum' variable to 1,2, or 3, respectively. 
