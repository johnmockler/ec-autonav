class MapObject:

    def __init__(self, coords):
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self,i):
        if i == 0:
            return self.coords.x
        elif i == 1:
            return self.coords.y
        else:
            return -1

    def __repr__(self):
        return 'Item({})'.format(self.coords)

class Arrow(MapObject):

    def __init__(self, coords, direction=None):
        super().__init__(coords)
        self.coords = coords
        self.direction = direction


    def __repr__(self):
        return 'Item({}, {})'.format(self.coords, self.direction)
    
class Line(MapObject):

    def __init__(self,coords,direct,segment):
        super().__init__(coords)
        self.coords = coords
        self.direct = direct
        self.segment = segment
    
    def __repr__(self):
        return 'Item({}, {}, {})'.format(self.coords, self.direct, self.segment)

class Obstacle(MapObject):

    def __init__(self,coords, radius=1):
        super().__init__(coords)
        self.coords = coords
        self.radius = radius

    def __repr__(self):
        return 'Item({}, {})'.format(self.coords, self.radius)