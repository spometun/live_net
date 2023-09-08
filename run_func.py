
import sys
import os
import importlib


file_path = sys.argv[1]
if file_path[-3:] != ".py":
    print(f"{file_path} not a python file")
    sys.exit(-1)

line_no = int(sys.argv[2])
print(sys.argv[1], sys.argv[2])
print(os.getcwd())
module_name = file_path[:-3].replace("/", ".")
print(module_name)
with open(file_path) as f:
    content = f.readlines()
line = content[line_no - 1]
if line[:4] != "def ":
    print(f"{line} not a python function")
    sys.exit(-1)

func_name = line[4:line.find("(")]
print(func_name)
module = importlib.import_module(module_name, module_name)
cmd = f"module.{func_name}()"
print(cmd)
exec(cmd)


# import 
