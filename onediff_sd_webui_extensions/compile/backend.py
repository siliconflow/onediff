from enum import Enum


class OneDiffBackend(Enum):
    ONEFLOW = "oneflow"
    NEXFORT = "nexfort"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"
