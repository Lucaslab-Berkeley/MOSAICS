# MOSAICS

MOSAICS is package for comparing where sets of particles in a cryo-EM agree/disagree with a reference template.

## Installation

The MOSAICS package can be installed via the pip package manager (*TODO: release to PyPi*) by using the pre-built wheel:
```
pip install mosaics
```
Or to install from source:
```
git clone https://github.com/Lucaslab-Berkeley/MOSAICS.git
cd MOSAICS
pip install -r requirements.txt
pip install .
```

More detailed instructions can be found on the documentation page.

## Building Documentation

Sphinx is used to manage and create the documentation for MOSAICS.
Building the documentation requires you to have MOSAICS installed locally with the additional docs requirements.
The documentation page can be build by running
```
pip install -r requirements-docs.txt
cd docs
make html
```