# PyEBSDIndex

Python based tool for Hough/Radon based EBSD indexing. The pattern processing is based on GPU processing. The processing is based on the work of S. I. Wright and B. L. Adams. Metallurgical Transactions a-Physical Metallurgy and Materials Science, 23(3):759–767, 1992 
And
N. Krieger Lassen. Automated Determination of Crystal Orientations from Electron Backscattering Pat- terns. PhD thesis, The Technical University of Denmark, 1994.

The band indexing is achieved through triplet voting using the methods outlined by A. Morawiec. Acta Crystallo- graphica Section A Foundations and Advances, 76(6):719–734, 2020.

Additionally NLPAR pattern processing is included (original distribution [NLPAR](https://github.com/USNavalResearchLaboratory/NLPAR) ; P. T. Brewick, S. I. Wright, and D. J. Rowenhorst. Ultramicroscopy, 200:50–61, May 2019.)



## Installation

The package only be installed from source at the moment:

```bash
git clone https://github.com/drowenhorst-nrl/PyEBSDIndex
cd PyEBSDIndex
pip install --editable .
```
