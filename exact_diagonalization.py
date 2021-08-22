#!/usr/bin/env python
# coding: utf-8

import numpy, sys
from inspect import getmembers, isfunction
import config
import initialization

#print(getmembers(initialization,isfunction))
config.Alpha = 1.
initialization.CreateLattice( nx=2, d=2, z=6 )
initialization.CreateEwald()
initialization.CreateBasis( config.ns / 2 )
print(config.Basis.shape)
