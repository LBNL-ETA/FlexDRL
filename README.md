# FlexDRL

## Setup and Execution

1. Pull the docker image: ``docker pull stouzani/drl_flexlab_v6:1``
2. if you are using mac use the original makefile; if you are using ubuntu then replace in the makefile `` -e DISPLAY=docker.for.mac.host.internal:0`` by `` -e DISPLAY=${DISPLAY} ``
3. Run the docker container: ``make run``
4. Run scripts:
    * test a simple control: ``python simulation/simulate.py``
    * test/run DRL training: ``python simulation/rl_train_ddpg.py``
    * test/run DRL prediction: ``python simulation/rl_predict_ddpg.py``

## Copyright

FlexDRL Copyright (c) 2021, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.


## License

FlexDRL is available under the following (license)[https://github.com/LBNL-ETA/FlexDRL/blob/main/LICENSE.txt].