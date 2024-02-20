# Neuroidal Model Implementation

Experiments on random graph models primarily drawn from the following paper:

   [Valiant, L. G. (2005). Memorization and Association on a Realistic Neural Model. *Neural Computation*, *17*, 527–555.](https://doi.org/10.1162/0899766053019890)

Code from this repository has been used in the following publications:

* Chowdhury, C. (December 2023). Foundations Of Memory Capacity In Models Of Neural Cognition. Master’s thesis, California Polytechnic State University, San Luis Obispo.

* Perrine, P. R. (June 2023). Neural Tabula Rasa: Foundations for Realistic Memories and Learning. Master's thesis. California Polytechnic State University, San Luis Obispo.

## Basic Instructions

1. Install miniconda or anaconda
2. Start a terminal in the Valiant directory
3. Create a virtual environment: `conda env create -f environment.yml`
4. To install new packages:
    * `conda install -n gt <packagename>`
    * Add them the YAML using `conda env export --no-builds > environment.yml`
5. Refresh the kernel list and select the `gt` kernel in the notebook.
6. For scripts, make sure you have the correct interpreter set in your editor.

### Sublist Model

1. Make sure Python and NumPy are installed
2. Run the ```sublists``` notebook cells
