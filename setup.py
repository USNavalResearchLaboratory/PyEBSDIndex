from setuptools import setup, find_packages

from pyebsdindex import __author__, __author_email__, __credits__, __description__, __name__, __version__


setup(
    # Package description
    name=__name__,
    version=__version__,
    license="Custom",
    python_requires=">=3.8",
    description=__description__,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Custom",
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
    download_url="https://github.com/USNavalResearchLaboratory/PyEBSDIndex",
    maintainer=__author__,
    maintainer_email=__author_email__,
    project_urls={
        "Bug Tracker": "https://github.com/USNavalResearchLaboratory/PyEBSDIndex/issues",
        "Source Code": "https://github.com/USNavalResearchLaboratory/PyEBSDIndex",
    },
    # Dependencies
    install_requires=[
        "matplotlib",
        "numpy",
        "numba",
        "pyopencl",
        "ocl_icd_wrapper_apple;sys_platform == 'darwin'",
        "ocl-icd-system;sys_platform == 'linux'",
        "pyswarms",
        "ray[default]",
        "scipy",
        "h5py",
        "jupyterlab"

    ],
    # Files to include when distributing package
    packages=find_packages(),
    package_dir={"pyebsdindex": "pyebsdindex"},
    include_package_data=True,
    package_data={
        "": ["License"],
        "pyebsdindex": ["*.py", "*.cl"],
    },
)
