import cv2
import numpy as np
from matplotlib import pyplot as plt
import math as m
from include import kdtree
import Arrow
import random

#MAP = r'C:\Users\jmock\Documents\Projekt Arbeit Images\ENC_test.png'

MAP = r'C:\Users\jmock\Documents\Projekt Arbeit Images\simpleTest.png'
TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])

#process map
map_image = cv2.imread(MAP,1)
hsv_img = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_threshed = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)


def detectArrows(mask):
    CORNER_THRESHOLD = 0.6
    #pre-process map before input to function  
    corners = cv2.goodFeaturesToTrack(mask,500,CORNER_THRESHOLD,10)
    corners = np.int0(corners)
    arrows = []

    for i in corners:
        x,y = i.ravel()
        arrow = Arrow.Arrow(x,y)
        arrows.append(arrow)
        #cv2.circle(map_image, (x,y),3,255,-1)

    #checks to see if any points are close to each other, and assumes they are detecting the same one and merges it
    arrows = filter_arrows(arrows)
    for arrow in arrows:
        cv2.circle(map_image,arrow.coords,3,255,-1 )
    #create kd tree of arrow objects
    #reference arrow object from search: tree.search_nn([1, 2])[0].data.next_node
    arrow_tree = kdtree.create(arrows)

    arrows = list(arrow_tree.inorder())
    #random.shuffle(arrows)
    return arrows,arrow_tree

def filter_arrows(arrows):
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
    d = m.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    return d

def midpoint(pt1,pt2):
    x = (pt1[0] - pt2[0])/2
    y = (pt1[1] - pt2[1])/2
    x = int(x + pt2[0])
    y = int(y + pt2[1])
    return (x,y)

def tree_search(node, tree, direct=None, layer=1):
    MAX_ANGLE = m.pi/18.0
    MAX_LAYER = 2
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

def check_path(node, direct):
    MAX_ANGLE = m.pi/12.0
    try:
        if node.data.next_node is None:
            return True
        elif node.data.fwd_dir-direct > MAX_ANGLE:
            return False
        else:
            return check_path(node.data.next_node, node.data.fwd_dir)
    except:
        if node.next_node is None:
            return True
        elif node.fwd_dir-direct > MAX_ANGLE:
            return False
        else:
            return check_path(node.next_node, node.fwd_dir)


#check the difference between two angles, which are in the range pi to -pi
#(pi minus -pi should be 0, as should 0 minus 0)
def angle_diff(theta1, theta2):
    diff = m.pi - abs(abs(theta1 - theta2)-m.pi)
    return diff


def direction(p1,p2):
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


def check_direction():
    #find closest arrow that the searched arrow doesnt share a path with
    #check whether it is on the left or right side of the fwd direction
    #

    ''' 
    - find N nearest neighbors
    - discount any that belong to same path
    - check that perpendicular distance > LIMIT and that the paths are almost parallel
    - check positive direction and see if line is on the left side. if not the direction is reverse (to go 'forward' on the path
    you need to to to the previous node...)
    - update line description

    '''
    return None

def main():
    # arrows,tree = detectArrows(map_mask)
    # for node in arrows:
    #     path,cost = tree_search(node,tree,3)
    # #   print(cost)         
    arrows,tree= detectArrows(map_mask)
    count = 0
    for arrow in arrows:
        cost, path = tree_search(arrow,tree)


    for arrow in arrows:
        if arrow.data.next_node is not None:
            cv2.line(map_image, (arrow.data.coords), (arrow.data.next_node.data.coords), (0, 255, 0), thickness=3, lineType=8)
            count += 1
        
    print(count)


    
    
    
    
    plt.imshow(map_image),plt.show()
    #tree_search1(arrows,tree)


main()