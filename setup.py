from setuptools import setup, find_packages
import os

def read(fname):
        return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='pysquid',
      version='0.0.1',
      description='Infer currents from flux images',
      long_description=read('README.md'),
      keywords='convex-optimization inference linear deconvolution squid',
      url='gligible.lassp.cornell.edu/colin/rnet',
      author='Colin Clement',
      author_email='colin.clement@gmail.com',
      license='GPLv3',
      packages=find_packages(exclude=['test*']),
      install_requires=['pyfftw', 'scipy', 'numpy', 'pypng'],
      python_requires='>=2.7, <4',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Medical Science Apps.'
      ]
     )
