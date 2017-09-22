import pygpu
ctx = pygpu.init('cuda')
array1 = pygpu.asarray([1,2,3], context=ctx)
array2 = pygpu.empty((3,), context=ctx)
array3 = pygpu.empty((3,), context=ctx)
print(array1)
print(array2)
print(array3)
