from setuptools import setup,find_packages

setup(
  name='bmel',
  version='0.1.0',
  packages=find_packages(include=['bmel', 'bmel.*']),
  python_requires='>=3.8',
  install_requires=['transformers',
                    'torch',
                    'mpi4py',
                    'boto3'
                   ]
)