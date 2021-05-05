import os
import setuptools

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
    name = "TwoPUtils",
    version = "0.0.1",
    author = "Giocomo Lab 2P Punks",
    author_email = "markplitt@gmail.com",
    description = ("General purpose two photon processing"),
    license = "BSD",
    keywords = "",
    url = "https://github.com/GiocomoLab/TwoPUtils",
    packages=setuptools.find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)