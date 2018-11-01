#!/usr/bin/env python
"""Python Casadi based Control and Estimation library

This is a library for control and estimation that employs
the Casadi framework for algorithm differentiation.
"""

from setuptools import setup, find_packages
import sys

import versioneer

if sys.version_info < (3, 5):
    raise SystemExit("requires  Python >= 3.5")

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Other
Topic :: Software Development
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# pylint: disable=invalid-name


setup(
    name='pyecca2',
    maintainer="James Goppert",
    maintainer_email="james.goppert@gmail.com",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    url='https://github.com/jgoppert/pyecca2',
    author='James Goppert',
    author_email='james.goppert@gmail.com',
    download_url='https://github.com/jgoppert/pyecca2',
    license='BSD 3-Clause',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    install_requires=[
        'scipy', 'numpy', 'casadi', 'pydot', 'matplotlib', 'pandas'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    #entry_points={
    #    'console_scripts': ['example=pyecca2.example:main'],
    #},
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
