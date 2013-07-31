Analysis of domestic power data.  Features include:

* Load power data from a variety of sources including:
  * MIT's REDD
  * Imperial Collect London's [UK Power Data
  (UKPD)](http://www.doc.ic.ac.uk/~dk3810/data/)
  * DECC / DEFRA's [Household Electricity Survey (HES)](http://randd.defra.gov.uk/Default.aspx?Menu=Menu&Module=More&Location=None&Completed=0&ProjectID=17359) dataset

## Requirements

* Cython
* Pandas

## Installation

Run `python setup.py build_ext --inplace` from the root of the pda directory structure (i.e. the directory containing `setup.py`)
