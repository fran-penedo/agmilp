from setuptools import setup, find_packages

config = {
    "description": "Assume-Guarantee Contract Mining Using MILP",
    "url": "",
    "author": "Fran Penedo",
    "author_email": "fran@franpenedo.com",
    "version": "1.1.0",
    "install_requires": [
        "numpy>=1.20.1",
        "gurobipy>=9.1.1",
        "scipy>=1.6.0",
        "templogic@git+git://github.com/fran-penedo/templogic.git@v2.0.4",
        "matplotlib>=3.3.4",
    ],
    "packages": find_packages(),
    "scripts": [],
    "entry_points": {},
    "name": "agmilp",
}

setup(**config)
