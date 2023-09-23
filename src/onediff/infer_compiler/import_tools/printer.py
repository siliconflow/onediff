class Printer:
    def __init__(self, color):
        self.color = color

    def __call__(self, msg):
        print(f"\033[{self.color}m{msg}\033[0m")


def colorize(color):
    def decorator(func):
        return Printer(color)

    return decorator


@colorize(31)
def print_red(msg):
    print(msg)


@colorize(32)
def print_green(msg):
    print(msg)


@colorize(37)
def print_white(msg):
    print(msg)


@colorize(33)
def print_yellow(msg):
    print(msg)


@colorize(34)
def print_blue(msg):
    print(msg)


if __name__ == "__main__":
    print_white("This is white text")
    print_red("This is red text")
    print_green("This is green text")
    print_yellow("This is yellow text")
    print_blue("This is blue text")
