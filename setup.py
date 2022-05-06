from itertools import chain
from setuptools import setup, find_packages

from pyebsdindex import (
    __author__, __author_email__, __credits__, __description__, __name__, __version__
)


# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
extra_feature_requirements = {
    "doc": [
        "furo",
        "nbsphinx >= 0.7",
        "sphinx >= 3.0.2",
        "sphinx-copybutton >= 0.2.5",
        "sphinx-gallery >= 0.6",
    ],
    "tests": ["coverage >= 5.0", "pytest >= 5.4", "pytest-cov >= 2.8.1"],
    "gpu": ["pyopencl"],
}
# Create a development project, including both the docs and tests projects
extra_feature_requirements["dev"] = list(
    chain(*list(extra_feature_requirements.values()))
)


setup(
    # Package description
    name=__name__,
    version=__version__,
    license="Custom",
    python_requires=">=3.8",
    description=__description__,
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "EBSD",
        "electron backscatter diffraction",
        "HI",
        "Hough indexing",
    ],
    zip_safe=True,
    # Contact
    author=__credits__,
    download_url="https://pypi.python.org/pypi/pyebsdindex",
    maintainer=__author__,
    maintainer_email=__author_email__,
    project_urls={
        "Bug Tracker": "https://github.com/USNavalResearchLaboratory/PyEBSDIndex/issues",
        "Documentation": "https://pyebsdindex.readthedocs.io",
        "Source Code": "https://github.com/USNavalResearchLaboratory/PyEBSDIndex",
    },
    url="https://pyebsdindex.readthedocs.io",
    # Dependencies
    extras_require=extra_feature_requirements,
    install_requires=[
        "h5py",
        "matplotlib",
        "numpy <= 1.21", # current requirement of numba
        "numba",
        "pyswarms",
        "ray[default]",
        "scipy",
    ],
    # Files to include when distributing package
    packages=find_packages(),
    package_dir={"pyebsdindex": "pyebsdindex"},
    include_package_data=True,
    package_data={
        "pyebsdindex": [
            "*.py",
            "*.cl",
            "tests/data/al_sim_20kv/al_sim_20kv.png",
        ],
    },
)
