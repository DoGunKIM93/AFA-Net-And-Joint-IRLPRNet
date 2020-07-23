from functools import reduce


def asd():
    print("?")


x = "resize(,,)"



func = x.split('(')[0]
#if '' in args : args.remove('')
args = list(filter(lambda y : y != '', x.split('(')[1][:-1].replace(' ','').split(',')))

#args = args.remove('5')

print(func)
print(args)

asd(*args)