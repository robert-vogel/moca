Method for Optimal Classification by Aggregation (MOCA)
===================================================



Dependencies
------------

MOCA has been validated with the following dependencies:

- Python (3.10)
- numpy (==1.26.3)
- scipy (==1.12.0)
- matplotlib (==3.8.0)
- summa (==1.0.0)


Installation
------------

Assuming that Python 3.10 is installed on your system, we'll
start by setting up a virtual environment named `env` and activating

```
python3.10 -m venv env
source env/bin/activate
```

The prompt on the terminal should update after activation so
that it is prefaced with `(env)`.  If not a mistake has occured.

```
python -m pip install --upgrade pip
```

Now install the `summa` package from the `pySUMMA` GitHub
repository.  If the dependencies `numpy`, `scipy`, and `matplotlib`
are not already installed, they will automatically be installed 
with `summa`

```
python -m pip install git+https://github.com/robert-vogel/pySUMMA.git
```

Next, and lastly, install `moca` from the GitHub repo:

```
python -m pip install git+https://github.com/robert-vogel/moca.git
```

Example
-------
