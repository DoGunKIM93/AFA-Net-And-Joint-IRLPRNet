from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec
import os

#pyModuleStrList = list(x[:-3] for x in os.listdir('.') if x.endswith('.py')) + list(f'backbone.{x[:-3]}' for x in os.listdir('./backbone') if x.endswith('.py')) 

spec = spec_from_file_location('version', 'model.py')
module = module_from_spec(spec)
spec.loader.exec_module(module)



#pyModuleObjList = list(map(import_module, pyModuleStrList))



#pyModuleStrList
#pyModuleObjList
#any(print(x) for x in pyModuleStrList)
#mdl = importlib.import_module(moduleList[0])

#mdl.version
#dict(x for x in inspect.getmembers(backbone.preprocessing) if (not x[0].startswith('__')) and (inspect.isclass(x[1])) )