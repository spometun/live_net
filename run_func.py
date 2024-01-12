
import sys
import os
import importlib
import json
import shutil


def update_old_args(file_path: str, line_no: int):
    launch_path = f"{os.getcwd()}/.vscode/launch.json"
    launch_path_backup = f"{os.getcwd()}/.vscode/launch.json.backup"
    shutil.copyfile(launch_path, launch_path_backup)

    with open(launch_path) as f:
        content = json.load(f)
    # print(content["configurations"])
    func = [el for el in content["configurations"] if el["name"] == "Func"]
    assert len(func) == 1, "Couldn't find launch configuration with name 'Func'"
    args = func[0]["args"].split()
    assert len(args) == 4
    args[2] = file_path
    args[3] = str(line_no)
    func[0]["args"] = " ".join(args)
    # print(func[0]["args"])
    # print(content)
    with open(launch_path, "w") as f:
        json.dump(content, f, indent = 4)
        
        
def try_run(file_path: str, line_no: int) -> bool:
    if file_path[-3:] != ".py":
        # print(f"{file_path} not a python file")
        return False

    # print(sys.argv[1], sys.argv[2])
    # print(os.getcwd())
    module_name = file_path[:-3].replace("/", ".")
    # print(module_name)
    with open(file_path) as f:
        content = f.readlines()
    line = content[line_no - 1]
    if line[:4] != "def ":
        # print(f"{line} not a python function")
        return False

    func_name = line[4:line.find("(")]
    # print(func_name)
    module = importlib.import_module(module_name, module_name)
    cmd = f"module.{func_name}()"
    # print(cmd)
    exec(cmd)
    return True
   

if __name__ == "__main__":
    file_path = sys.argv[1]
    line_no = int(sys.argv[2])
    file_path_old = sys.argv[3]
    line_no_old = int(sys.argv[4])
    ok = try_run(file_path, line_no)
    if ok:
        update_old_args(file_path, line_no)
    else:
        try_run(file_path_old, line_no_old)



