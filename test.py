from fast_slic import Slic
from fast_slic.avx2 import SlicAvx2
from PIL import Image
import numpy as np

from matplotlib import pyplot


img = Image.open("m.png")

r = Slic(num_components=256, min_size_factor=0).iterate(np.array(img))
print(r)
'''
i = Image.fromarray(r)

print(i)

i.save("r.png")
'''


pyplot.imsave('r.png', r)