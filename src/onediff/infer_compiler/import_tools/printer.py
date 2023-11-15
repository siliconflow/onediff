""" Utility Module """
import warnings

class Printer:
    #printing text with colored output.
    def __init__(self, color):
        self.color = color

    def __call__(self, *args, **kwargs):
        return f"\033[{self.color}m" + f"{args} {kwargs}" + "\033[0m"


def print_red(*args, **kwargs) -> None:
    # Print text with red color.
    output = Printer(31)(*args, **kwargs)
    warnings.warn(output)


def print_green(*args, **kwargs) -> None:
    #Print text with green color.
    output = Printer(32)(*args, **kwargs)
    # warnings.warn(output)
    print(output)


def print_yellow(*args, **kwargs) -> None:
    #Print text with yellow color.
    output = Printer(33)(*args, **kwargs)
    print(output)


if __name__ == "__main__":
    print_red("This is red text")
    print_green("This is green text")
