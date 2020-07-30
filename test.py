import itertools


a = [[1,2,3],[4,5,6]]
b = [['a','b','c'],['a','b','c']]

d = list( map( lambda x: dict(zip(['dataFilePath','labelFilePath'], x)), list(zip(list(itertools.chain.from_iterable(a)),list(itertools.chain.from_iterable(b)) ))))
print(d)
