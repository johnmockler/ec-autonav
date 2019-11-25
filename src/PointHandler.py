import math as m

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return 2
    
    
def distance(pt1,pt2):
    '''calculates distance between two points
    Parameters
    ----------
    pt1,pt2: point objects
    
    '''
    return m.sqrt((pt1.x-pt2.x)**2 + (pt1.y-pt2.y)**2)

def midpoint(pt1,pt2):
    '''calculates midpoint between two points
    Parameters
    ----------
    pt1,pt2: point objects

    Returns
    ----------

    point object
    '''
    x = (pt1.x - pt2.x)/2
    y = (pt1.y - pt2.y)/2
    x = int(x + pt2.x)
    y = int(y + pt2.y)
    
    return Point(x,y)

def direction(p1,p2):
    '''calculates direction between two points
    Parameters
    ----------
    p1, p2: tuple object with an x and y value
    '''
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    theta = m.atan2(dy,dx)
    return theta

def cross_product(ptA, ptB, ptP):
    '''calculates whether the point P  falls to the right of line segment AB.
    https://www.geeksforgeeks.org/direction-point-line-segment/

    ----------
    Parameters
    ptA: tuple of origin point (x,y)
    ptB: tuple of forward point in segment (x,y)
    ptP: tuple of test point (x,y)
    ----------
    Returns
    side: 1 if right, -1 if left, 0 if it is on line segment
    '''
    #set A as origin
    vx = ptB.x - ptA.x
    vy = ptB.y - ptA.y
    x = ptP.x - ptA.x
    y = ptP.y - ptA.y

    x_product = vx*y - vy*x

    return x_product

def closest_point(ptA, ptB, ptP):
    #http://paulbourke.net/geometry/pointlineplane/

    #rename for easier reading
    x1, y1 = ptA.x,ptA.y
    x2, y2 = ptB.x,ptB.y
    x3, y3 = ptP.x,ptP.y


    unum = (x3-x1)*(x2-x1) + (y3 - y1)*(y2 - y1)
    udenom = distance(Point(x1,y1),Point(x2,y2))**2

    u = unum/udenom

    x = x1 + u*(x2-x1)
    y = y1 + u*(y2-y1)

    return Point(x,y)


def perpendicular_dist(ptA, ptB, ptP):
#https://www.intmath.com/plane-analytic-geometry/perpendicular-distance-point-line.php
    #set A as origin
    vx = ptB.x - ptA.x
    vy = ptB.y - ptA.y

    px = ptP.x - ptA.x
    py = ptP.y - ptA.y

    #put line segment B in ax + by = 0 form

    #a = -y/x
    a = - vy/vx
    b = 1.0
    m = px
    n = py

    num = abs(a*m + b*n)
    denom = (a**2 + b**2)**0.5

    return num/denom

def scale_angle(theta):
    multiplier = int(theta/2)
    if theta > m.pi:
        theta = m.fmod(theta,m.pi)
        if multiplier%2 != 0:
            theta = -m.pi - theta
        else:
            theta = -theta
    elif theta < -m.pi:
        theta = m.fmod(theta,m.pi)

        if multiplier%2 != 0:
            theta = m.pi + theta

        else:
            theta = -theta
    return theta