from setuptools import setup, find_packages, Extension
from os.path import join

CPP_DIR = 'cpp'
CYTHON_DIR = 'cython'

try:
    # This trick adapted from 
    # http://stackoverflow.com/a/4515279/732596
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

if use_cython:
    sources = [join(CYTHON_DIR, 'load_pwr_data.pyx'),
               join(CPP_DIR, 'load_pwr_data_cpp.cpp')]

    extensions = [Extension("pda.load_pwr_data", 
                            sources=sources,
                            include_dirs=['cpp']
                           )]

    ext_modules = cythonize(extensions, language="c++")
else:
    ext_modules = [
        Extension("pda.load_pwr_data", 
                  [join(CYTHON_DIR, 'load_pwr_data.cpp')]),
    ]

setup(
    name='pda',
    version='0.1',
    packages = find_packages(),
    install_requires = ['numpy', 'pandas'],
    description='Power Data Analytics',
    author='Jack Kelly',
    author_email='jack@jack-kelly.com',
    url='https://github.com/JackKelly/pda',
    download_url = "https://github.com/JackKelly/pda/tarball/master#egg=powerstats-dev",
    long_description=open('README.md').read(),
    license='MIT',
    ext_modules=ext_modules,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='smartmeters power electricity energy analytics redd'
)
