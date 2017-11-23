#!/usr/bin/env python
import sys
from operator import itemgetter, attrgetter, methodcaller

class Sorter:
	def __init__(self, s):
		alphamask = ''
		for c in s:
			if c.isalnum():
				alphamask += '1'
			else:
				alphamask += '0'
		self.lower = s.lower()
		self.mask = alphamask
		self.data = s
		

if len(sys.argv) == 2:
	lines = []
	with open(sys.argv[1]) as f:
		for line in f:
			lines.append(Sorter(line.rstrip()))
	lines.sort(key=attrgetter('lower', 'mask', 'data'))
	for line in lines:
		print(line.data)
