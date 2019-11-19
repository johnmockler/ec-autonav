import math as m
from include import kdtree

#testing - - - - - - - - - - - - - - - - - - - - - - - -
import cv2
import numpy as np
from matplotlib import pyplot as plt




MAP = r'C:\Users\jmock\Documents\Projekt Arbeit Images\simpleTest.png'
TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])

#process map
map_image = cv2.imread(MAP,1)
hsv_img = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_threshed = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)
CORNER_THRESHOLD = 0.6
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Arrow:

    def __init__(self, x, y, next_node=None, prev_node = None, fwd_dir = None, bwd_dir = None, line = None):
        self.coords = (x,y)
        self.next_node = next_node
        self.prev_node = prev_node
        self.fwd_dir = fwd_dir
        self.bwd_dir = bwd_dir
        self.line = line

    def __len__(self):
        return len(self.coords)

    def __getitem__(self,i):
        return self.coords[i]

    def __repr__(self):
        return 'Item({}, {}, {}, {}, {}, {}, {})'.format(self.coords[0], self.coords[1], self.next_node, self.prev_node, self.fwd_dir, self.bwd_dir, self.line)

class Line:

    def __init__(self):
        self.lineArray = []
        self.direction = None
        self.name = ''
    
    def __getitem__(self,i):
        return self.lineArray[i]

    def add_element(self, node):
        self.lineArray.append(node)

    def get_path(self):
        return self.lineArray

    def set_direction(self, direction):
        self.direction = direction

    def append(self, item):
        self.lineArray.append(item)

    def give_name(self,nameString):
        self.name = 'Line ' + nameString


#@loa
def load_arrows(potential_arrows):
    '''Loads a list of potential arrows into a kd_tree for easy searching and returns as a list of KD_Node objects.
    The first element in the list is the root of the tree.

    Parameters
    ----------
    potential_arrows: list of x,y coordinates of found arrows.

    Returns
    -------
    arrows: list of KD_node objects

    '''

    arrows = []
    for i in potential_arrows:
        x,y = i.ravel()
        arrow = Arrow(x,y)
        arrows.append(arrow)
    
    arrows = filter_arrows(arrows)
    arrow_tree = kdtree.create(arrows)
    arrows = list(arrow_tree.inorder())

    return arrows, arrow_tree

def tree_search(node, tree, direct=None, layer=1):
    ''' Searches a node on a given tree and attempts to find a path which minimizes the angle between nodes. i.e. attempts to find 
    a set of nodes in a straight line

    Parameters
    ----------
    node: KD_node object to be searched
    tree: KD_node object which is the root of tree being searched
    direct: direction of previous node. not necessary for user input
    layer: layer being searched. not necessary for user input

    Returns
    -------
    best cost: int of the calculated cost for the found path
    best path: a list of kd_node objects that make up the found path
    '''

    MAX_ANGLE = m.pi/18.0
    MAX_LAYER = 3
    MAX_DISTANCE = 4500.0
    COST_MODIF = 0.6
    best_path = []
    best_cost = 1000
    best_dir = 0
    pre_path = []

    if layer == 1:
        if node.data.prev_node is not None:
            direct = node.data.bwd_dir
            pre_path = [(node,0)]
        else:
            pre_path = [(node,0)]

        if node.data.next_node is not None:
            return best_cost,best_path

    next_layer = tree.search_knn(node.data.coords, 6)

    if layer < MAX_LAYER:
        for branch_node in next_layer:
            curr_layer = layer
            fwd_cost = None
            temp_cost = 1000

            if (branch_node[1] != 0 and branch_node[1] < MAX_DISTANCE) and branch_node[0].data.prev_node is None:

                branch_direct = direction(node.data.coords, branch_node[0].data.coords)
                temp_path = [(branch_node[0],branch_direct)]

                if direct is not None:
                    delta = angle_diff(branch_direct,direct)
                else:
                    delta = None
                
                
                if is_valid_direction(branch_direct) and ((delta != None and abs(delta) <= MAX_ANGLE) or delta == None):

                    if branch_node[0].data.next_node is None:
                        curr_layer += 1
                        fwd_cost, fwd_path = tree_search(branch_node[0], tree, branch_direct, curr_layer)
                        
                    else:
                        curr_layer += 2
                        fwd_delta = angle_diff(branch_node[0].data.fwd_dir, branch_direct)*COST_MODIF

                        if abs(fwd_delta) <= MAX_ANGLE:
                            fwd_cost, fwd_path = tree_search(branch_node[0].data.next_node, tree, branch_node[0].data.fwd_dir, curr_layer)
                            if len(fwd_path)>0:
                                fwd_cost = (abs(fwd_delta)+abs(fwd_cost)/len(fwd_path))
                            else:
                                fwd_cost = fwd_delta
                        
                    if fwd_cost is not None:
                        if delta == None:
                            temp_cost = fwd_cost
                            temp_path = temp_path + fwd_path
                        else:
                            if len(fwd_path)>0:
                                temp_cost = (abs(fwd_cost) + abs(delta))/(len(fwd_path)+1)
                            else:
                                temp_cost = abs(delta)
                            temp_path = temp_path + fwd_path
                        if temp_cost < best_cost:
                            best_cost = temp_cost
                            best_path = temp_path
                            best_dir = branch_direct
    
        best_path = pre_path + best_path

        if layer == 1:
            for i in range(len(best_path)-1):
                best_path[i][0].data.next_node = best_path[i+1][0]
                best_path[i][0].data.fwd_dir = best_path[i+1][1]
                best_path[i+1][0].data.prev_node = best_path[i][0]
                best_path[i+1][0].data.bwd_dir = best_path[i+1][1]

        return best_cost, best_path

    else:
        for branch_node in next_layer:
            temp_cost = 1000
        #consider allowing better nodes to replace...
            if (branch_node[1] != 0 and branch_node[1] < MAX_DISTANCE) and branch_node[0].data.prev_node is None:
                branch_direct = direction(node.data.coords,branch_node[0].data.coords)
                #delta = branch_direct - direct
                delta = angle_diff(branch_direct,direct)
                temp_path = [(branch_node[0],branch_direct)]
                temp_cost = abs(delta)
                
                if is_valid_direction(branch_direct):
                    if branch_node[0].data.next_node is None:

                        if temp_cost<best_cost and temp_cost<=MAX_ANGLE:
                            best_cost = temp_cost
                            best_dir = branch_direct
                            best_path = temp_path
                            

                    elif abs(branch_node[0].data.fwd_dir - branch_direct)*COST_MODIF<=MAX_ANGLE:
                        if temp_cost<best_cost and temp_cost<=MAX_ANGLE:
                            best_path = temp_path
                            best_cost = temp_cost
                            best_dir = branch_direct

        if len(best_path)>0 and layer == 1:   
            node.data.next_node = best_path[0][0]
            best_path[0][0].data.prev_node = node
            node.data.fwd_dir = best_dir
            best_path[0][0].data.bwd_dir = best_dir

        best_path = pre_path + best_path
        return best_cost, best_path

def filter_arrows(arrows):
    '''takes a list of points, finds arrows within the Minimum distance and replaces them with their midpoint
    Parameters
    ----------
    arrows: list of arrow objects

    Returns
    -------
    arrows: list list of arrow objects
    '''

    MIN_DISTANCE = 17.0

    for arrow in arrows:
        for next_arrow in arrows:
            dist = distance(arrow.coords, next_arrow.coords)
            if dist != 0 and dist < MIN_DISTANCE:
                arrow.coords = midpoint(arrow.coords,next_arrow.coords)
                arrows.remove(next_arrow)
    print(len(arrows))
    return arrows
        
def distance(pt1,pt2):
    '''calculates distance between two points
    Parameters
    ----------
    pt1,pt2: tuple object with an x and y value
    
        '''
    d = m.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    return d

def midpoint(pt1,pt2):
    '''calculates midpoint between two points
    Parameters
    ----------
    pt1,pt2: tuple object with an x and y value
    '''
    x = (pt1[0] - pt2[0])/2
    y = (pt1[1] - pt2[1])/2
    x = int(x + pt2[0])
    y = int(y + pt2[1])
    return (x,y)

def angle_diff(theta1, theta2):
    '''calculates difference between two angles
    Parameters
    ----------
    theta1, theta2: int angle in radians
    '''
    diff = m.pi - abs(abs(theta1 - theta2)-m.pi)
    return diff

def direction(p1,p2):
    '''calculates direction between two points
    Parameters
    ----------
    p1, p2: tuple object with an x and y value
    '''
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    theta = m.atan2(dy,dx)
    return theta

#checks if it is a valid direction to build a node
def is_valid_direction(theta):
    if theta < m.pi/4.0 and theta > -m.pi/4.0:
        return True
    elif theta > m.pi/4.0 and theta < 3*m.pi/4.0:
        return False
    elif theta > 3*m.pi/4.0 and theta < -3*m.pi/4.0:
        return False
    elif theta > -3*m.pi/4.0 and theta < -m.pi/4.0:
        return True

def group_lines(arrows):
    color = 0
    name = 0

    for arrow in arrows:
        if arrow.data.line is None:
            prev = arrow.data.prev_node
            nxt = arrow.data.next_node
            line = Line()
            arrowSet = [arrow]
            while prev is not None:
                arrowSet.append(prev)
                prev = prev.data.prev_node
            
            while nxt is not None:
                arrowSet.append(nxt)
                nxt = nxt.data.next_node
            for item in arrowSet:
                item.data.line = line
                line.append(item)
                line.give_name(str(name))
            name += 1

    return arrows

def left_or_right(ptA, ptB, ptP):
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
    ptB[0] -= ptA[0]
    ptB[1] -= ptA[1]
    ptP[0] -= ptA[0]
    ptP[1] -= ptA[1]

    x_product = ptB[0]*ptP[1] - ptB[1]*ptP[0]

    if x_product > 0:
        side = 1
    elif x_product <0:
        side = -1
    else:
        side = 0
    return side

def determine_line_direction(arrow, tree):
    MIN_ANGLE = m.pi/8.0
    neighbors = search_knn(arrow, tree, 4)
    neighbor_fwd = True
    arrow_fwd = True
    for node in neighbors:
        if node.data.line != arrow.data.line:

            ptA = arrow.data.coords
            if arrow.data.next_node is not None:
                ptB = arrow.data.next_node.data.coords
                arrow_fwd = True
            elif arrow.data.prev_node is not None:
                ptB = arrow.data.prev_node.data.coords
                arrow_fwd = False
            else:
                raise ValueError('Node is not connected to path')
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #make sure lines are not close to parallel
            if True:

                if left_or_right(ptA, ptB, ptP) <= 0 and arrow_fwd:
                    arrow.data.line = 1
                    if neighbor_fwd:
                        node.data.line = 1
                    else:
                        node.data.line = -1

                elif left_or_right(ptA, ptB, ptP) < 0 and arrow_fwd:
                    arrow.data.line = -1 
                    if neighbor_fwd:
                        no



                # if delta >= 0 and arrow_fwd:
                #     if neighbor_fwd:
                #         arrow.data.line = 1
                #         node.data.line = 1
                #     else:
                #         arrow.data.line = 1
                #         node.data.line = -1
                # elif delta >= 0 and not arrow_fwd:
                #     if neighbor_fwd:
                #         arrow.data.line = -1
                #         node.data.line = -1
                #     else:
                #         arrow.data.line = -1
                #         node.data.line = 1
                # elif delta <= 0 and arrow_fwd:
                #     if neighbor_fwd:
                #         arrow.data.line = -1
                #         node.data.line = -1
                #     else:
                #         arrow.data.line = -1
                #         node.data.line = 1   
                # elif delta <= 0 and not arrow_fwd:
                #     if neighbor_fwd:
                #         arrow.data.line = -1
                #         node.data.line = -1
                #     else:
                #         arrow.data.line = -1
                #         node.data.line = 1   
            return  

def set_line_directions(arrows,tree):
    for arrow in arrows:
        if arrow.data.line is None:
            try:
                determine_line_direction(arrow,tree)
            except ValueError:
                pass

if __name__ == "__main__":
    corners = cv2.goodFeaturesToTrack(map_mask,500,CORNER_THRESHOLD,10)
    corners = np.int0(corners)

    arrows, tree = load_arrows(corners)
    for arrow in arrows:
        cost, path = tree_search(arrow,tree)


    arrows = group_lines(arrows)

    # for arrow in arrows:
    #     if arrow.data.next_node is not None:
    #         cv2.line(map_image, (arrow.data.coords), (arrow.data.next_node.data.coords), (0, 255, 0), thickness=3, lineType=8)    
    plt.imshow(map_image),plt.show()