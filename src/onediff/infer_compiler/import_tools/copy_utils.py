from typing import List, Union
from pathlib import Path
import os
import ast
import subprocess
import shutil
import fnmatch

FUNC_PREFIX = "ProxyFunc"
FUNC_SUFFIX = "Of"

try:
    import astor
except ImportError as e:
    subprocess.run(["pip", "install", "astor"])

import astor

__all__ = ["copy_files", "get_matched_files", "convert_funcs_to_classes", "get_proxy_func_name"]

def get_proxy_func_name(name):
    return f"{FUNC_PREFIX}{name}{FUNC_SUFFIX}"

def get_matched_files(
    root: Union[str, Path], ignore_file=".gitignore", extra_ignore_rules=["*setup.py"]
):
    ignore_rules = []
    ignore_file = Path(root) / ignore_file
    if ignore_file.exists():
        with open(ignore_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignore_rules.append(line)

    ignore_rules.extend(extra_ignore_rules)
    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            filepath = os.path.relpath(filepath, root)
            if len(ignore_rules) == 0:
                matches.append(filepath)
                continue
            is_match = any(fnmatch.fnmatch(filepath, rule) for rule in ignore_rules)
            if not is_match:
                matches.append(filepath)

    return matches


def copy_files(src_dir, dst_dir, filelist):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    for file in filelist:
        src = src_dir / file
        dst = dst_dir / file

        if not dst.parent.exists():
            dst.parent.mkdir(parents=True)

        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"{src} does not exist!")



def parse_file(file_path: Union[str, Path]) -> ast.AST:
    with open(file_path, "r") as f:
        code = f.read()
    return ast.parse(code)


def find_function_nodes(tree: ast.AST) -> List[ast.FunctionDef]:
    """Find all function nodes not in class definition in the tree"""
    funcs = []
    class_funcs = set()
    source_code = astor.to_source(tree)
    first_def_funcs = set()
    for line in source_code.split("\n"):
        if line.startswith("def"):
            first_def_funcs.add(line.split(" ")[1].split("(")[0].strip())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for node in ast.walk(node):
                if isinstance(node, ast.FunctionDef):
                    class_funcs.add(node)

        if isinstance(node, ast.FunctionDef):
            funcs.append(node)

    return [func for func in funcs if func not in class_funcs and func.name in first_def_funcs]


def generate_class_definition(node):
    class_name = node.name
    class_docstring = ast.Expr(
        value=ast.Str(s=f"Auto generated class for {class_name} by onediff")
    )
    class_def = ast.ClassDef(
        name=f"{FUNC_PREFIX}{class_name}{FUNC_SUFFIX}",
        bases=[],
        keywords=[],
        body=[
            class_docstring,
            ast.FunctionDef(                
                name=class_name, 
                args=node.args, 
                body=node.body, 
                decorator_list=[ast.Name(id='staticmethod', ctx=ast.Load())]
            ),
        ],
        decorator_list=[],
    )
    return class_def


def classdef_to_python(classdef):
    return astor.to_source(classdef)


def convert_funcs_to_classes(input_file: Union[str,Path], output_file: Union[str,Path]):
    """Convert functions into classes of the same name automatically"""
    tree = parse_file(input_file)
    source_code = astor.to_source(tree)
    function_nodes = find_function_nodes(tree)
    for node in function_nodes:
        class_def = generate_class_definition(node)
        source_code += classdef_to_python(class_def)
    
    with open(output_file, "w") as f:
        f.write(source_code)

if __name__ == "__main__":
    input_file = "test_fake_torch/_fake_torch.py"  # 输入Python文件名
    output_file = "output.py"  # 输出新的Python文件名

    convert_funcs_to_classes(input_file, output_file)
    print("Done!")
