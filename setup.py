import sys
import subprocess

PY_VER = sys.version[0]
subprocess.call(["pip{:} install -r requirements.txt".format(PY_VER)], shell=True)

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='transly',
      version='0.1.0',
      description='Transliteration, Hindi to English and English to Hindi',
      url='https://github.com/gitnik17/transly',
      author='Nikhil Kothari',
      author_email='gitnik17@gmail.com',
      license='Apache License 2.0',
      zip_safe=False,
      setup_requires=[
          'pandas',
          'keras==2.3.1',
          'tensorflow==1.15.2',
      ],
      install_requires=[
          'pandas',
          'keras==2.3.1',
          'tensorflow==1.15.2',
      ],
      packages=['transly'],
      package_data={'transly': ['transly/*']},
      include_package_data=True,
      classifiers=[
          "Programming Language :: Python :: 3"
      ],
      long_description = long_description
      )
