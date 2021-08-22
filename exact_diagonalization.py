#!/usr/bin/env python
# coding: utf-8

print("hello world")
print("inserted in insert mode in vim")
print("want to make a branch")

from inspect import getmembers, isfunction
import initialization

print(getmembers(initialization,isfunction))

global InvDist
InvDist = initialization.CreateEwald(6)

initialization.CreateLattice( 9, 1, 2 )
