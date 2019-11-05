class Arrow:

    #arrows are represented as points, which lie on the center point of the actual arrows
    #and directions
    def __init__(self, x, y, next_node = None, prev_node = None, fwd_dir = None, bwd_dir = None):
        self.coords = (x,y)
        self.next_node = next_node
        self.prev_node = prev_node
        self.fwd_dir = fwd_dir
        self.bwd_dir = bwd_dir

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self,i):
        return self.coords[i]
    
    def __repr__(self):
        return 'Item({}, {}, {}, {}, {},{})'.format(self.coords[0], self.coords[1], self.next_node, self.prev_node, self.fwd_dir, self.bwd_dir)
    
 
class Path:

    def __init__(self):
        self.path_list = ()
        self.deviation = 0
    
    def add_arrow(self, arrow):
        self.path_list.append(arrow)

    

