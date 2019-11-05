class ArrowObject():

    def __init__(self, x, y, nxt=None, prev=None, dst_nxt = 0, dst_prev = 0):
        self.x = x
        self.y = y
        self.nxt = nxt
        self.dst_nxt = dst_nxt
        self.prev = prev
        self.dst_prev = dst_prev

    
