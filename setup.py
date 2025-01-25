#! /usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages



with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ThermaNewt',
    version='0.0.1',
    url='https://github.com/michaelremington2/ThermaNewt',
    author='Michael Remington and Jeet Sukumaran',
    author_email='michaelremington2@gmail.com',
    license="LICENSE.txt",
    classifiers=[
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    ],
    # scripts=[
    #     "src/rattle_newton/sim_snake_tb.py",
    #     "src/rattle_newton/thermal_summary_stats.py",
    #     ],
    test_suite = "tests",
    package_dir={"": "src"},
    description="Ectotherm body temperature simulator using newtons law of cooling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    python_requires=">=3.8",   
)