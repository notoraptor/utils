from __future__ import print_function, absolute_import, division
import sys

def resolve(v1, v2, v3, v4):
	a = ((v1 - v2 - v4 + v3), 2)
	b = ((v1 + v2 + v4 - v3), 2)
	c = ((v3 - v1 + v2 + v4), 2)
	d = ((v3 - v1 - v2 + v4), 2)
	return (a, b, c, d)

def frac_string(f):
	return "%s/%s" % f

if len(sys.argv) == 5:
	v1 = int(sys.argv[1])
	v2 = int(sys.argv[2])
	v3 = int(sys.argv[3])
	v4 = int(sys.argv[4])
	a, b, c, d = resolve(v1, v2, v3, v4)
	print(
"""
%(a)s	+	%(b)s	=	%(v1)s
+		+
%(c)s	-	%(d)s	=	%(v2)s
=		=
%(v3)s		%(v4)s
""" % dict(v1=v1, v2=v2, v3=v3, v4=v4, a=frac_string(a), b=frac_string(b), c=frac_string(c), d=frac_string(d)))