from setuptools import setup

setup(
  name='bmel',
  version='0.1.0',
  packages=find_packages(),
  python_requires='>=3.8',
  install_requires=['transformers',
                    'torch',
                    'mpi4py'
                   ]
)