class Arrow:

    #arrows are represented as points, which lie on the center point of the actual arrows
    #and directions
<<<<<<< Updated upstream
    def __init__(self, x, y, direct):
        self.x = x
        self.y = y
        self.direct = direct

=======
    def __init__(self, x, y, next_node=None, prev_node = None, fwd_dir = None, bwd_dir = None):
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
        return 'Item({}, {}, {}, {}, {}, {})'.format(self.coords[0], self.coords[1], self.next_node, self.prev_node, self.fwd_dir, self.bwd_dir)
    

    
>>>>>>> Stashed changes

