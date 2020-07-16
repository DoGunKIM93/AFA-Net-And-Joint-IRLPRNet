
def dup(input):
    from collections import Counter
    from functools import reduce

    tmp = reduce(lambda x, y: x & y,  list(map( lambda x: Counter(x), input  ))  )

    return reduce(lambda x, y : x + y ,list(list(s for p in range(int(tmp[s]))) for s in tmp))

print(dup(["hanwha", "haha", "valueha"]))
