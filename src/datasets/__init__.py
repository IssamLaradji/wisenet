from os.path import dirname, basename, isfile
import glob
from importlib import import_module

fnameList = glob.glob(dirname(__file__)+"/*.py")
fnameList += glob.glob(dirname(__file__)+"/*/*.py")


# module = import_module("{}.{}".format(module_name, mName))

__all__ = []
for fname in fnameList:
    if fname.endswith('__init__.py'):
        continue
    i = fname.find("datasets")

    mname = fname[i:].replace(".py","").replace("/",".")
    __all__ += [import_module(mname)]



