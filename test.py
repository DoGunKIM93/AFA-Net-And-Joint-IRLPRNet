


dataType = {'a':'text-float', 'b':'image-int'}


a = dict([key, dict(zip(['dataName', 'dataType', 'tensorType'], [key] + dataType[key].split('-') ))] for key in dataType if (dataType[key].split('-')[0] in ['text', 'image', 'imageSequence'] and dataType[key].split('-')[1] in ['float', 'int', 'long', 'double']))

print(a)
