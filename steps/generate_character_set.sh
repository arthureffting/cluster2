#!/bin/bash

# Runs a python script wrapped in the arguments [same arguments for all scripts]
/bin/bash steps/utils/bash_wrap.sh ${@} --script=generate_character_set.py
