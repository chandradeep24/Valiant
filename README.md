# Valiant

Experiments on random graphs

### Instructions

Fork the repo before cloning. Submit pull requests to have your changes merged.

1. Install python@3.9 or higher with pip (recommended)
2. Create a virtual environment: `python3 -m venv venv`
3. Activate the environment: `source venv/bin/activate`
4. Install the required packages: `pip install -r requirements.txt`
5. Figure out a way to install graph tool seperately and import it into the python installation in the virtual env. Note that it will install its own numpy and scipy dependencies, you need to figure out a way to make it use the numpy and scipy in the venv.
6. Add new packages that you use to the requirements.txt using `pip freeze requirements.txt`
