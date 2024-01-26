import os

from setuptools import setup, find_packages

version = '0.0.1'

setup(
    name='argus',
    version=version,
    packages=find_packages(),
    package_dir={'': '.'},
    include_package_data=True,
    zip_safe=False,
    requires=[]
)