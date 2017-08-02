from __future__ import absolute_import, print_function, division
import sys, argparse

def get_infos_from_header(header, results):
	sep = '+' * 70
	prefix = 'ERROR: '
	start = 0
	header_len = len(header)
	start_header = 0
	if results and results[0] is header:
		start += 1
	for i in range(start, len(results)):
		result = results[i]
		for line in result:
			if line:
				if line.startswith(prefix):
					result_id = line[len(prefix):]
					if result_id:
						header_infos = []
						count = 0
						for j in range(start_header, header_len):
							count += 1
							header_line = header[j]
							if header_line.startswith(result_id) or header_infos:
								header_infos.append(header_line)
								if header_line.endswith(prefix[:-2]):
									break
						if header_infos:
							results[i] = header_infos + [sep] + result
						start_header += count
				break

def collect(results, include, exclude, outputs):
	for result in results:
		result_string = '\n'.join(result)
		if all(s in result_string for s in include) and not any(s in result_string for s in exclude):
			outputs.append(result_string)

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Path to "nosetests -vs" output file.')
parser.add_argument('-i', '--include', action='append', default=[], help='Words or strings that should be in a result text to select.')
parser.add_argument('-x', '--exclude', action='append', default=[], help='Words or strings that should NOT be in a result text to select.')
parser.add_argument('-c', '--count', action='store_true', default=False, help='Print number of total and selected results.')
parser.add_argument('-f', '--first', action='store_true', default=False, help='Print header (lines before the first error result).')
parser.add_argument('-F', '--include-first', action='store_true', default=False, help='Consider header (lines before the first error result) as a result too.')
parser.add_argument('-n', '--no-print', action='store_true', default=False, help='Do not print selected results.')
parser.add_argument('-l', '--parse-header', action='store_true', default=False, help='Try to get infos from header for every result.')

args = parser.parse_args(sys.argv[1:])

header = None
all_results = []
current_lines = []
with open(args.filename, 'r') as file:
	for line in file:
		line = line.strip()
		len_line = len(line)
		if len_line > 0 and line == '=' * len_line:
			all_results.append(current_lines)
			current_lines = []
		else:
			current_lines.append(line)

all_results.append(current_lines)

if len(all_results) > 0:
	first_result = all_results[0]
	if first_result and any(line for line in first_result):
		header = first_result

if header is None or args.include_first:
	results = all_results
else:
	results = all_results[1:]

if header and args.parse_header:
	get_infos_from_header(header, results)

outputs = []
if args.include or args.exclude:
	collect(results, args.include, args.exclude, outputs)
if args.count:
	len_results = len(results)
	len_output = len(outputs)
	print(len_results, 'result%s' % ('' if len_results < 2 else 's'))
	if len_output:
		print(len_output, 'selected%s' % ('' if len_output < 2 else 's'))
if args.first and header:
	print('\n'.join(header))
if not args.no_print:
	for output in outputs:
		print('=' * 70)
		print(output)
