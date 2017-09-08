from __future__ import absolute_import, print_function, division

class Aligner:

	match_score = 1
	diff_score = -1
	indel_score = -1
	indel_symbol = ' '

	def __init__(self, match=1, mismatch=-1, indel=-1, gap='-'):
		self.match_score = match
		self.diff_score = mismatch
		self.indel_score = indel
		self.indel_symbol = gap

	def similarity(self, a, b):
		if ord(a) == ord(b):
			return self.match_score
		elif a == ' ' or b == self.indel_symbol:
			return self.indel_score
		else:
			return self.diff_score

	def align(self, A, B, debug=False):
		# A in first column
		# B in first line
		A = self.indel_symbol + A
		B = self.indel_symbol + B
		matrix_width = len(B)
		matrix_height = len(A)
		matrix = [[0] * matrix_width for _ in range(matrix_height)]
		for i in range(matrix_width):
			matrix[0][i] = i * self.diff_score
		for i in range(1, matrix_height):
			matrix[i][0] = i * self.diff_score
		for i in range(1, matrix_height):
			for j in range(1, matrix_width):
				match = matrix[i-1][j-1] + self.similarity(A[i], B[j])
				deletion = matrix[i-1][j] + self.indel_score
				insertion = matrix[i][j-1] + self.indel_score
				matrix[i][j] = max(match, deletion, insertion)
		alignment_A = ''
		alignment_B = ''
		alignment_diff = ''
		i = matrix_height - 1
		j = matrix_width - 1
		while i > 0 or j > 0:
			if i > 0 and j > 0 and matrix[i][j] == matrix[i-1][j-1] + self.similarity(A[i], B[j]):
				alignment_A = A[i] + alignment_A
				alignment_B = B[j] + alignment_B
				alignment_diff = ('X' if A[i] == B[j] else '*') + alignment_diff
				i = i-1
				j = j-1
			elif i > 0 and matrix[i][j] == matrix[i-1][j] + self.indel_score:
				alignment_A = A[i] + alignment_A
				alignment_B = self.indel_symbol + alignment_B
				alignment_diff = '-' + alignment_diff
				i = i-1
			else:
				alignment_A = self.indel_symbol + alignment_A
				alignment_B = B[j] + alignment_B
				alignment_diff = '-' + alignment_diff
				j = j-1
		if debug:
			for row in matrix:
				if row:
					print(row[0], end='')
					for i in range(1, len(row)):
						print('\t%s' % row[i], end='')
				print()
		return (alignment_A, alignment_B, alignment_diff)

A = "theano.gpuarray.tests.check_dnn_conv.TestDnnConv2D.test_fwd('time_once', 'float16', 'float16', ((2, 3, 300, 5), (2, 3, 40, 4), (1, 1), (1, 1), 'half', 'conv', 2.0, 0)) ... (using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM (timed), ws:0, hash:FWD|GPU#0000:06:00.0 -t -g 1 -dim 2,3,300,5,4500,1500,5,1 -filt 2,3,40,4 -mode conv -pad 20,2 -subsample 1,1 -dilation 1,1 -hh [unaligned])"
B = "theano.gpuarray.tests.check_dnn_conv.TestDnnConv2D.test_fwd('time_once', 'float16', 'float16', ((2, 3, 300, 5), (2, 3, 40, 4), (1, 1), (1, 1), 'half', 'conv', 2.0, 0)) ... (using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM (timed), ws:0, hash:FWD|GPU#0000:06:00.0 -t -g 1 -dim 2,3,300,5,4500,1500,5,1 -filt 2,3,40,4 -mode conv -pad 20,2 -subsample 1,1 -dilation 1,1 -hh [unaligned])"
alignment = Aligner().align(A, B, debug=False)
for x in alignment:
	print(x)