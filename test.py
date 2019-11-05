from include import kdtree

# This class emulates a tuple, but contains a useful payload
class Item(object):
    def __init__(self, x, y, data1, data2=None):
        self.coords = (x, y)
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i]

    def __repr__(self):
        return 'Item({}, {}, {}, {})'.format(self.coords[0], self.coords[1], self.data1, self.data2)

# Now we can add Items to the tree, which look like tuples to it
point1 = Item(2, 3, 'First', 'xx')
point2 = Item(3, 4, 'Second',point1)
point3 = Item(5, 2, ['some', 'list'])

# Again, from a list of points
tree = kdtree.create([point1, point2, point3])

#  The root node
print(tree)

# ...contains "data" field with an Item, which contains the payload in "data" field
print(tree.data.data2.data1)

# All functions work as intended, a payload is never lost
print(tree.search_nn([1, 2])[0].data.data1)