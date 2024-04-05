import os
from setuptools import setup
from setuptools import find_packages

setup(name='dr_submodular',
      author='',
      version='0.0.1',
      description='DR submodular algorithms',
      license='MIT',
      install_requires=[
            'matplotlib==3.7.1',
            'numpy==1.24.2',
            'cvxopt==1.3.0',
            'scipy==1.10.1'
      ],
      packages=find_packages(
            include=['dr_submodular',
            'dr_submodular.quadratic_programming',
            'dr_submodular.revenue_maximization']
      ))

