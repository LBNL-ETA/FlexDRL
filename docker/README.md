This directory contains files for building docker images used for co-simulation of
the DER project.  The docker images are split into four as follows:

``der_base`` builds from Docker's ``ubuntu:16.04``, installs basic software and
python packages, and creates a new user: developer.

``der_jmodelica`` builds from ``der_base`` and installs JModelica for co-simulation.

``der_eplus:8.5`` builds from ``der_jmodelica`` and installs EnergyPlus 8.5.

To build the images:

1. ``$ cd base`` then ``$ make build``.  Takes around 8 mins.
2. ``$ cd jmodelica`` then ``$ make build``.  Takes around 60 mins.
3. ``$ cd eplus`` then ``$ make build``.  Takes around 3 mins.

To run the test script:

1. ``$ cd eplus`` then ``$ make run``
2. ``$ git clone https://github.com/samirtouzani/FlexLab_Toy_Model.git`` to download 
the updated script then ``$ cd FlexLab_Toy_Model``
3. ``$ python test_xxx.py`` to run the test script 

