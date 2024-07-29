import collections


class LRUCache(collections.OrderedDict):
    __slots__ = ["LEN"]

    def __init__(self, capacity: int = 9):
        self.LEN = capacity

    def get(self, key: str, default=None) -> any:
        if key in self:
            self.move_to_end(key)
            return self[key]
        else:
            return default

    def put(self, key: str, value: any) -> None:
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.LEN:
            self.popitem(last=False)
