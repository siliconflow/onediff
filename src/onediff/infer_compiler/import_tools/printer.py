import sys
from pathlib import Path


class Printer:
    def __init__(self, color):
        self.color = color

    def __call__(self, *args, **kwargs):
        print(f"\033[{self.color}m", end="")
        print(*args, **kwargs)
        print("\033[0m")
       

def colorize(color):
    def decorator(func):
        return Printer(color)

    return decorator


@colorize(31)
def print_red(*args, **kwargs):
    print(*args, **kwargs)


@colorize(32)
def print_green(*args, **kwargs):
    print(*args, **kwargs)


@colorize(37)
def print_white(*args, **kwargs):
    print(*args, **kwargs)


@colorize(33)
def print_yellow(*args, **kwargs):
    print(*args, **kwargs)


@colorize(34)
def print_blue(*args, **kwargs):
    print(*args, **kwargs)



if __name__ == "__main__":
    print_white("This is white text", "This is also white text", {"a": 1})
    print_red("This is red text")
    print_green("This is green text")
    print_yellow("This is yellow text")
    print_blue("This is blue text")
