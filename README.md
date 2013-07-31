# Power Data Analysis (PDA) toolkit

Analysis of domestic power data.  Features include:

* Load power data from a variety of sources including:
  * MIT's [REDD](http://redd.csail.mit.edu/)
  * Imperial Collect London's [UK Power Data
  (UKPD)](http://www.doc.ic.ac.uk/~dk3810/data/)
  * DECC / DEFRA's [Household Electricity Survey
  (HES)](http://randd.defra.gov.uk/Default.aspx?Menu=Menu&Module=More&Location=None&Completed=0&ProjectID=17359)
  dataset
* Correlate appliance activity with weather data
* Create appliance usage histograms for average days, weeks or months
* Histograms of appliance on-durations
* Histograms of appliance power consumption
* Calculate total energy usage per time period
* Lots of other stuff...

## Requirements

* Cython
* Pandas

## Installation

Run `python setup.py build_ext --inplace` from the root of the pda directory structure (i.e. the directory containing `setup.py`)

## Documentation

Please note that this code is not fantastically well documented (yet)!
All documentation is in the code. Some functions have doc strings.

## Directory structure

    ├── `cpp`: C++ implementation and header files written by a human (!)
    |
    ├── `cython`: Cython source files (pyx) and C++ files generated by Cython
    |
    ├── `pda`: library code
       |
       ├── `tests`: unittest code
           |
           ├── `test_data`
