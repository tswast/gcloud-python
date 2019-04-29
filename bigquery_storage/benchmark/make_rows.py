
class Row(object):
    __slots__ = ["some_tuple"]

    def __init__(self, some_tuple):
        self.some_tuple = some_tuple

rows = []
for i in range(5552452):
    rows.append(Row(()))
    #rows.append(())
