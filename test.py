
import backbone.augmentation
import backbone.preprocessing
import inspect

AUGMENTATION_DICT = dict(x for x in inspect.getmembers(backbone.augmentation) if (not x[0].startswith('_')) and (inspect.isfunction(x[1])) )
PREPROCESSING_DICT = dict(x for x in inspect.getmembers(backbone.preprocessing) if (not x[0].startswith('__')) and (inspect.isclass(x[1])) )

print(AUGMENTATION_DICT)