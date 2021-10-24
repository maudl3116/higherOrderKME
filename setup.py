import os
import re
import setuptools
from Cython.Build import cythonize

here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'higherOrderKME', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

ext_modules = [
    setuptools.Extension(
        name="cython_backend",
        sources=["higherOrderKME/sigkernel/cython_backend.pyx"],
    )
]

setuptools.setup(
    name="higherOrderKME",
    version=version,
    author="Maud Lemercier",
    author_email="maud.lemercier@warwick.ac.uk",
    description="Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes in PyTorch.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/maudl3116/higherOrderKME",
    packages=setuptools.find_packages(exclude=['examples', 'tests']),
    install_requires=["cython >= 0.29", "numba >= 0.50", "torch >= 1.6.0", "numpy==1.20.1", "scipy==1.6.2", 
                      "imageio==2.9.0", "pymunk==6.0.0", "h5py==2.8.0", "matplotlib==3.3.4", "Pillow==8.2.0", 
                      "pyflex==0.1.5", "fbm==0.3.0"], 
    python_requires='~=3.6',
    ext_modules=cythonize(ext_modules), 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
