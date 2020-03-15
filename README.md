# hamiltonianlearning

This repository contains a numpy implementation of the Hamiltonian learning framework outlined in https://arxiv.org/abs/1911.12548

To be brief, the code maximizes the likelihood that the measured data was produced by a Hamiltonian via a descent method.

I hope to add a cleanly written Tensorflow implementation that can run on a GPU in the future. This is not a top priority for me at the moment. It would allow for testing on huge hamiltonian matrices.

The code requires NumPy, SciPy, and Python3. Save both files (hamiltonianlearning.py and hlFunctions.py) to the same folder and run the file hamiltonianlearning.py. This will automatically simulate the learning of a random Hamiltonian matrix. The code will run in its entirety in a few seconds. You can change the seed value (or remove that line enitrely) to simulate a different Hamiltonian, or you can change the number of quantum states.

By uncommenting a few lines you can also try to learn the Hamiltonian for hyperfine splitting of hydrogen or the cyclobutadiene molecule.

There are still a number of optizations that could be made. I wish I had not explicitly defined the basis for the Hamiltonian matrices, because this is not memory efficient. I will fix this in the Tensorflow gpu version.
