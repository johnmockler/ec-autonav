import cv2
import numpy as np
from matplotlib import pyplot as plt
import math as m
from include import kdtree
from src import Arrow
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
    #pre-process map before input to function  
    corners = cv2.goodFeaturesToTrack(mask,500,0.6,10)
    corners = np.int0(corners)
    arrows = []

    for i in corners:
        x,y = i.ravel()
        arrow = Arrow.Arrow(x,y)
        arrows.append(arrow)
        cv2.circle(map_image, (x,y),3,255,-1)


    #create kd tree of arrow objects
    #reference arrow object from search: tree.search_nn([1, 2])[0].data.next_node
    arrow_tree = kdtree.create(arrows)

    arrows = list(arrow_tree.inorder())
    random.shuffle(arrows)
    return arrows,arrow_tree

def tree_search(node, tree, direct=None, layer=1):
    MAX_ANGLE = m.pi/14.0
    #MAX_ANGLE = 1000
    MAX_LAYER = 3
    best_path = []
    best_cost = 1000
    best_dir = 0
    pre_path = []
    #run initial code
    #create path here????

    if layer == 1:
        if node.data.prev_node is not None:
            direct = node.data.bwd_dir
            pre_path = [(node,0)]
        else:
            pre_path = [(node,0)]

        if node.data.next_node is not None:
            print('here')
            return best_cost,best_path

    next_layer = tree.search_knn(node.data.coords, 5)

    if layer < MAX_LAYER:
        for branch_node in next_layer:
            curr_layer = layer
            fwd_cost = None
            temp_cost = 1000
            #use node here or branch_node????

            #if there is a previous node, it tries to compare everything to "None.data"...
            if branch_node[1] != 0 and branch_node[0].data.prev_node is None:

                branch_direct = direction(node.data.coords, branch_node[0].data.coords)
                temp_path = [(branch_node[0],branch_direct)]

                if direct is not None:
                    delta = angle_diff(branch_direct,direct)
                else:
                    delta = None
                
                
                if (delta != None and abs(delta) <= MAX_ANGLE) or delta == None:

                    if branch_node[0].data.next_node is None:
                        curr_layer += 1
                        fwd_cost, fwd_path = tree_search(branch_node[0], tree, branch_direct, curr_layer)
                        
                    else:
                        curr_layer += 2
                        fwd_delta = angle_diff(branch_node[0].data.fwd_dir, branch_direct)

                        if abs(fwd_delta) <= MAX_ANGLE:
                            fwd_cost, fwd_path = tree_search(branch_node[0].data.next_node, tree, branch_node[0].data.fwd_dir, curr_layer)
                            if len(fwd_path)>0:
                                fwd_cost = abs(fwd_delta)+abs(fwd_cost)/len(fwd_path)
                            else:
                                fwd_cost = fwd_delta
                            #fwd_path = [branch_node[0].data.next_node] + fwd_path
                        
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

        #WE SHOULD ONLY UPDATE THE NODES WITH NEXT NODE AND SUCH AT THE LAST LAYER...
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
            if branch_node[1]!=0 and branch_node[0].data.prev_node is None:
                branch_direct = direction(node.data.coords,branch_node[0].data.coords)
                #delta = branch_direct - direct
                delta = angle_diff(branch_direct,direct)
                temp_path = [(branch_node[0],branch_direct)]
                temp_cost = abs(delta)

                if branch_node[0].data.next_node is None:

                    if temp_cost<best_cost and temp_cost<=MAX_ANGLE:
                        best_cost = temp_cost
                        best_dir = branch_direct
                        best_path = temp_path
                        

                elif abs(branch_node[0].data.fwd_dir - branch_direct)<=MAX_ANGLE:
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
        if arrow.data.next_node is None:
            cost,path = tree_search(arrow,tree)
            #cv2.line(map_image, (arrow.data.coords), (arrow.data.next_node.data.coords), (0, 255, 0), thickness=3, lineType=8)

        else:
            cv2.line(map_image, (arrow.data.coords), (arrow.data.next_node.data.coords), (0, 255, 0), thickness=3, lineType=8)
            try:
                print('fwd dir' + str(arrow.data.fwd_dir))
                print('bwd dir' + str(arrow.data.bwd_dir))
                print(arrow.data.fwd_dir - arrow.data.bwd_dir)
            except:
                print('oof')



    
    
    
    
    plt.imshow(map_image),plt.show()
    #tree_search1(arrows,tree)


main()