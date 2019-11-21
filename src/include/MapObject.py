class MapObject:
    def __init__(self, x, y):
        self.coords = (x,y)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self,i):
        return self.coords[i]

    def __repr__(self):
        return 'Item({}, {})'.format(self.coords[0], self.coords[1])

class Arrow(MapObject):

    def __init__(self, x, y, next_node=None, prev_node = None, fwd_dir = None, bwd_dir = None, line = None):
        super().__init__(x,y)
        self.coords = (x,y)
        self.next_node = next_node
        self.prev_node = prev_node
        self.fwd_dir = fwd_dir
        self.bwd_dir = bwd_dir
        self.line = line


    def __repr__(self):
        return 'Item({}, {}, {}, {}, {}, {}, {})'.format(self.coords[0], self.coords[1], self.next_node, self.prev_node, self.fwd_dir, self.bwd_dir, self.line)
    
class Line(MapObject):

    def __init__(self,x1,y1,x2,y2):
        super().__init__(x,y)
        self.coords = midpoint((x1,y1),(x2,y2))
        self.direction = direction((x1,y1),(x2,y2))
    
    def __repr__(self):
        return 'Item({}, {}, {})'.format(self.coords[0], self.coords[1], self.direction)