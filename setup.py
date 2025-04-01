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
        "nbsphinx                       >= 0.7",
        "numpydoc",
        "pydata-sphinx-theme",
        "sphinx                         >= 3.0.2",
        "sphinx-codeautolink[ipython]",
        "sphinx-copybutton              >= 0.2.5",
        "sphinx-design",
        "sphinx-gallery",
    ],
    "tests": [
        "coverage                       >= 5.0",
        "pytest                         >= 5.4",
        "pytest-cov                     >= 2.8.1",
    ],
    "gpu": [
        "pyopencl",
    ],
    "parallel": [
        "ray[default]                   >= 2.9",
    #    "pydantic                       < 2",
    ]
}
# Create a development installation "dev" including "doc" and "tests"
# projects
extra_feature_requirements["dev"] = list(
    chain(*list(extra_feature_requirements.values()))
)
# Create a user installation "all" including "gpu" and "parallel"
runtime_extras_require = {}
for x, packages in extra_feature_requirements.items():
    if x not in ["doc", "tests"]:
        runtime_extras_require[x] = packages
extra_feature_requirements["all"] = list(chain(*list(runtime_extras_require.values())))


setup(
    # Package description
    name=__name__,
    version=__version__,
    license="Custom",
    python_requires=">=3.9",
    description=__description__,
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
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
        "Radon indexing",
        "NLPAR",
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
        "numpy",
        "numba>=0.55",
        "scipy",
    ],
    # Files to include when distributing package (see also MANIFEST.in)
    packages=find_packages(),
    package_dir={"pyebsdindex": "pyebsdindex"},
    include_package_data=True,
)
