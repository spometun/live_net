from life.lib.simple_log import LOG
import os
import sys

a = 3
b = 4
print(f"hi {a + b}")
print(sys.path)
print(f'env = {os.environ.get("PYTHONPATH")}') 
print("/home/spometun/projects/home_project/life/testvs.py:3")
print("life/testvs.py:3")
# LOG("here")