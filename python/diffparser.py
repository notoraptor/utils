from __future__ import absolute_import, print_function, division
import sys
import re
from aligner import Aligner

if len(sys.argv) != 2:
	exit(-1)

class Difference(object):
	def __init__(self, diff_mode, from_start, from_end, to_start, to_end, from_lines, to_lines):
		self.diff_mode = diff_mode
		self.from_start = int(from_start)
		self.from_end = int(from_end)
		self.to_start = int(to_start)
		self.to_end = int(to_end)
		self.from_lines = from_lines
		self.to_lines = to_lines
		assert self.diff_mode in ('a', 'c', 'd'), self.diff_mode
		assert all(l.startswith('<') for l in self.from_lines), '\n'.join(self.from_lines)
		assert all(l.startswith('>') for l in self.to_lines), '\n'.join(self.to_lines)

	def _interval_str(self, a, b):
		return '%d' % a if a == b else '%d,%d' % (a, b)

	def diff_pos_str(self):
		return '%s%s%s' % (self._interval_str(self.from_start, self.from_end), self.diff_mode, self._interval_str(self.to_start, self.to_end))

	def is_stable_change(self):
		return self.diff_mode == 'c' and len(self.from_lines) == len(self.to_lines)

	def smart_print(self):
		aligner = Aligner(match=1, indel=-1, mismatch=-1)
		print( '%s%s%s' % (self._interval_str(self.from_start, self.from_end), self.diff_mode, self._interval_str(self.to_start, self.to_end)) )
		if diff_mode == 'd':
			for line in self.from_lines:
				print(line)
		if diff_mode == 'a':
			for line in self.to_lines:
				print(line)
		if diff_mode == 'c':
			if len(self.from_lines) == len(self.to_lines):
				alignments_list = []
				alignments_set = set()
				for i in range(len(self.from_lines)):
					A = self.from_lines[i]
					B = self.to_lines[i]
					A = '' if A == '<' else A[2:]
					B = '' if B == '>' else B[2:]
					alignment = aligner.align(A, B)
					characteristics = alignment.unique_difference()
					if characteristics is None:
						characteristics = alignment.strip_similarities()
					if characteristics is None:
						print(alignment)
					else:
						alignments_list.append(characteristics)
						alignments_set.add(characteristics)
				if len(alignments_set) == 1:
					if len(self.from_lines) > 1:
						print('[All same]')
					for l in alignments_list[0]:
						print(l)
					return alignments_list[0]
				else:
					for al in alignments_list:
						for l in al:
							print(l)
			else:
				for line in self.from_lines:
					print(line)
				print('---')
				for line in self.to_lines:
					print(line)
		print()
		return None

# a: add
# c: change
# d: delete
pattern_diff_start = re.compile('^([0-9]+)(,([0-9]+))?([acd])([0-9]+)(,([0-9]+))?$')

differences = []
additions = []
deletions = []
stable_changes = []
unstable_changes = []

with open(sys.argv[1], 'r') as diff_file:
	diff_mode = ''
	from_start = 0
	from_end = 0
	to_start = 0
	to_end = 0
	from_lines = []
	to_lines = []
	accumulate_from = True
	for line in diff_file:
		line = line.strip()
		matcher = pattern_diff_start.match(line)
		if matcher:
			if diff_mode:
				difference = Difference(diff_mode, from_start, from_end, to_start, to_end, from_lines, to_lines)
				differences.append(difference)
				if diff_mode == 'a':
					additions.append(difference)
				elif diff_mode == 'c':
					if difference.is_stable_change():
						stable_changes.append(difference)
					else:
						unstable_changes.append(difference)
				elif diff_mode == 'd':
					deletions.append(difference)
			from_start, from_end, diff_mode, to_start, to_end = matcher.group(1, 3, 4, 5, 7)
			if from_end is None:
				from_end = from_start
			if to_end is None:
				to_end = to_start
			from_lines = []
			to_lines = []
			if diff_mode == 'a':
				accumulate_from = False
			elif diff_mode == 'c':
				accumulate_from = True
			elif diff_mode == 'd':
				accumulate_from = True
		elif line == '---':
			accumulate_from = not accumulate_from
		elif accumulate_from:
			from_lines.append(line)
		else:
			to_lines.append(line)
	difference = Difference(diff_mode, from_start, from_end, to_start, to_end, from_lines, to_lines)
	differences.append(difference)
	if diff_mode == 'a':
		additions.append(difference)
	elif diff_mode == 'c':
		if difference.is_stable_change():
			stable_changes.append(difference)
		else:
			unstable_changes.append(difference)
	elif diff_mode == 'd':
		deletions.append(difference)

map_differences = dict()

print(len(differences), 'difference' + ('s.' if len(differences) != 1 else '.'))
print(len(unstable_changes), 'unstable change' + ('s.' if len(unstable_changes) != 1 else '.'))
print(len(stable_changes), 'stable change' + ('s.' if len(stable_changes) != 1 else '.'))
print(len(additions), 'addition' + ('s.' if len(additions) != 1 else '.'))
print(len(deletions), 'deletion' + ('s.' if len(deletions) != 1 else '.'))
for name, elements in zip(('Additions', 'Deletions', 'Unstable changes', 'Stable changes'), (additions, deletions, unstable_changes, stable_changes)):
	if elements:
		print('=' * (len(name) + 1))
		print(name)
		print('=' * (len(name) + 1))
		for element in elements:
			characteristics = element.smart_print()
			if characteristics is not None:
				a, b, diff = characteristics
				if a > b:
					a, b = b, a
				characteristics = a, b, diff
				if characteristics not in map_differences:
					map_differences[characteristics] = []
				map_differences[characteristics].append(element)

if map_differences and any(len(l) > 1 for l in map_differences.values()):
	print()
	print('=============================')
	print('Some differences categorized:')
	print('=============================')
	for c, l in sorted(map_differences.items(), key = lambda x : len(x[1])):
		print(len(l), 'difference%s with:' % ('' if len(l) == 1 else 's'))
		for ll in c:
			print('\t%s' % ll)
		for d in l:
			print(d.diff_pos_str())
		print()
