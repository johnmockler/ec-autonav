import cv2
import numpy as np
from matplotlib import pyplot as plt
import math as m
from include import kdtree
import Arrow
import random

MAP = r'C:\Users\jmock\Documents\Projekt Arbeit Images\ENC_test.png'

#MAP = r'C:\Users\jmock\Documents\Projekt Arbeit Images\simpleTest.png'
TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])

#process map
map_image = cv2.imread(MAP,1)
hsv_img = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_threshed = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)


def detectArrows(mask):
    CORNER_THRESHOLD = 0.5
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
    arrows = group_arrows(arrows)
    for arrow in arrows:
        cv2.circle(map_image,arrow.coords,3,255,-1 )
    #create kd tree of arrow objects
    #reference arrow object from search: tree.search_nn([1, 2])[0].data.next_node
    arrow_tree = kdtree.create(arrows)

    arrows = list(arrow_tree.inorder())
    random.shuffle(arrows)
    return arrows,arrow_tree

def group_arrows(arrows):
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
    MAX_ANGLE = m.pi/16.0
    #MAX_ANGLE = 1000
<<<<<<< Updated upstream:ArrowSearchTest.py
    MAX_LAYER = 2
=======
    MAX_LAYER = 4
    MAX_DISTANCE = 500.0
>>>>>>> Stashed changes:src/Pathmatching.py
    best_path = []
    best_cost = 1000
    best_dir = 0
    pre_path = []
    #run initial code
    #create path here????

    if layer == 1:
        if node.data.prev_node is not None:
            direct = node.data.bwd_dir
            pre_path = [node]
        else:
            pre_path = [node]

        if node.data.next_node is not None:
            return best_cost,best_path

    next_layer = tree.search_knn(node.data.coords, 5)

    if layer < MAX_LAYER:
        for branch_node in next_layer:
            curr_layer = layer
            fwd_cost = None
            temp_cost = 1000
            #use node here or branch_node????

            #if there is a previous node, it tries to compare everything to "None.data"...
            if (branch_node[1] != 0 or branch_node[1] > MAX_DISTANCE) and branch_node[0].data.prev_node is None:

                branch_direct = direction(node.data.coords, branch_node[0].data.coords)
                temp_path = [branch_node[0]]

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

        node.data.next_node = branch_node[0]
        branch_node[0].data.prev_node = node
        node.data.fwd_dir = best_dir
        branch_node[0].data.bwd_dir = best_dir
        return best_cost, best_path

    else:
        for branch_node in next_layer:
            temp_cost = 1000
        #consider allowing better nodes to replace...
            if (branch_node[1] != 0 or branch_node[1] > MAX_DISTANCE) and branch_node[0].data.prev_node is None:
                branch_direct = direction(node.data.coords,branch_node[0].data.coords)
                #delta = branch_direct - direct
                delta = angle_diff(branch_direct,direct)
                temp_path = [branch_node[0]]
                temp_cost = abs(delta)

                if branch_node[0].data.next_node is None:

                    if temp_cost<best_cost and temp_cost<=MAX_ANGLE:
                        best_path = temp_path
                        best_cost = temp_cost
                        best_dir = branch_direct

                elif abs(branch_node[0].data.fwd_dir - branch_direct)<=MAX_ANGLE:
                #elif check_path(branch_node[0].data,branch_direct):
                    if temp_cost<best_cost and temp_cost<=MAX_ANGLE:
                        best_path = temp_path
                        best_cost = temp_cost
                        best_dir = branch_direct
        #ERROR IS HERE!!!
        if len(best_path)>0:   
            node.data.next_node = best_path[0]
            best_path[0].data.prev_node = node
            node.data.fwd_dir = best_dir
            best_path[0].data.bwd_dir = best_dir

        best_path = pre_path + best_path
        return best_cost, best_path

def tree_search2(arrows,arrow_tree):
    MAX_ANGLE = m.pi/8.0
    for root in arrows:
        #check if it already has a next node
        if root.data.next_node is None:
            #high initial score
            path_cost = 1000
            path = []
            direct = []
            #search for the first 6 nearest neighbors to root node
            first_layer = arrow_tree.search_knn(root.data.coords,5)

            root_dir = None
            

            #check if root node already has a previous, if so use in calculation
            if root.data.prev_node is not None: 
                root_dir = root.data.bwd_dir
        
            
            for first_node in first_layer:
                #check if the first node is already linked to another or if the first node is actually the root.
                try:
                    if first_node[1]!=0 and first_node[0].data.prev_node==None:
    
                        first_dir = direction(root.data.coords, first_node[0].data.coords)

                        if root_dir!=None:
                            delta1 = first_dir-root_dir
                        else:
                            delta1 = None

                        # #verify that the backward angle difference is not too great
                        # try:
                        #     if abs(first_dir - root_dir)>MAX_AN

                        if (delta1!=None and abs(delta1)<=MAX_ANGLE) or delta1==None:
                
                            if first_node[0].data.next_node==None:
                                second_layer = arrow_tree.search_knn(first_node[0].data.coords,6)

                                for second_node in second_layer:
                                    #make sure second node isnt already linked or if it was the first node, or if it was the original node
                                    if second_node[0].data.prev_node==None and second_node[1]!=0 and second_node[0].data.coords!=root.data.coords:
                                        second_dir = direction(first_node[0].data.coords,second_node[0].data.coords)
                                        delta2 = second_dir-first_dir
                                        if delta1 is not None:
                                            temp_cost = mean(abs(delta2),abs(delta1))
                                        else:
                                            temp_cost = abs(delta2)
                                        if temp_cost<abs(path_cost) and abs(delta2)<=MAX_ANGLE:
                                            path = [root, first_node[0], second_node[0]]
                                            #path = [root, first_node[0]]
                                            path_cost=temp_cost
                                            direct = (first_dir, second_dir)

                                            third_layer = arrow_tree.search_knn(first_node[0].data.coords,6)
                            else:
                                
                                second_dir = first_node[0].fwd_dir
                                delta2 = second_dir-first_dir
                                temp_cost = abs(delta2)+abs(delta1)
                                print(temp_cost)
                                if temp_cost<abs(path_cost) and abs(delta)<=MAX_ANGLE:
                                    path = [root, first_node[0], first_node[0].next_node]
                                    path_cost = temp_cost
                                    #path = [root, first_node[0]]
                                    direct = (first_dir, second_dir)
                            

                except:
                    next  
            
            if len(path)==3:
                root.data.next_node = path[1]
                root.data.fwd_dir = direct[0]
                path[1].data.prev_node = root
                path[1].data.bwd_dir = direct[0]
                path[1].data.next_node = path[2]
                path[1].data.fwd_dir = direct[1]
                path[2].data.prev_node = path[1]
                path[2].data.bwd_dir = direct[1]
                #print(root.data.coords)
                #print(path[1].data.coords)
                #print(path[2].data.coords)

                #print(direct)
                #print(direct[0]-direct[1])
                try:
                    print(path)
                except:
                    next
                                    
                cv2.line(map_image, (path[0].data.coords), (path[1].data.coords), (0, 255, 0), thickness=3, lineType=8)
                
                cv2.line(map_image, (path[1].data.coords), (path[2].data.coords), (0, 255, 0), thickness=3, lineType=8)
    plt.imshow(map_image),plt.show()

def tree_search1(arrows,arrow_tree):
    MAX_ANGLE = m.pi/8.0
    for root in arrows:
        #check if it already has a next node
        if root.data.next_node is None:
            #high initial score
            path_cost = 1000
            path = []
            direct = []
            #search for the first 6 nearest neighbors to root node
            first_layer = arrow_tree.search_knn(root.data.coords,5)

            root_dir = None
            

            #check if root node already has a previous, if so use in calculation
            if root.data.prev_node is not None: 
                root_dir = root.data.bwd_dir
        
            
            for first_node in first_layer:
                #check if the first node is already linked to another or if the first node is actually the root.
                try:
                    if first_node[1]!=0 and first_node[0].data.prev_node==None:
    
                        first_dir = direction(root.data.coords, first_node[0].data.coords)

                        if root_dir!=None:
                            delta1 = first_dir-root_dir
                        else:
                            delta1 = None

                        # #verify that the backward angle difference is not too great
                        # try:
                        #     if abs(first_dir - root_dir)>MAX_AN

                        if (delta1!=None and abs(delta1)<=MAX_ANGLE) or delta1==None:
                
                            if first_node[0].data.next_node==None:
                                second_layer = arrow_tree.search_knn(first_node[0].data.coords,6)

                                for second_node in second_layer:
                                    #make sure second node isnt already linked or if it was the first node, or if it was the original node
                                    if second_node[0].data.prev_node==None and second_node[1]!=0 and second_node[0].data.coords!=root.data.coords:
                                        second_dir = direction(first_node[0].data.coords,second_node[0].data.coords)
                                        delta2 = second_dir-first_dir
                                        if delta1 is not None:
                                            temp_cost = mean(abs(delta2),abs(delta1))
                                        else:
                                            temp_cost = abs(delta2)
                                        if temp_cost<abs(path_cost) and abs(delta2)<=MAX_ANGLE:
                                            path = [root, first_node[0], second_node[0]]
                                            #path = [root, first_node[0]]
                                            path_cost=temp_cost
                                            direct = (first_dir, second_dir)
                            else:
                                
                                second_dir = first_node[0].fwd_dir
                                delta2 = second_dir-first_dir
                                temp_cost = abs(delta2)+abs(delta1)
                                print(temp_cost)
                                if temp_cost<abs(path_cost) and abs(delta)<=MAX_ANGLE:
                                    path = [root, first_node[0], first_node[0].next_node]
                                    path_cost = temp_cost
                                    #path = [root, first_node[0]]
                                    direct = (first_dir, second_dir)
                            

                except:
                    next  
            
            if len(path)==3:
                root.data.next_node = path[1]
                root.data.fwd_dir = direct[0]
                path[1].data.prev_node = root
                path[1].data.bwd_dir = direct[0]
                path[1].data.next_node = path[2]
                path[1].data.fwd_dir = direct[1]
                path[2].data.prev_node = path[1]
                path[2].data.bwd_dir = direct[1]
                #print(root.data.coords)
                #print(path[1].data.coords)
                #print(path[2].data.coords)

                #print(direct)
                #print(direct[0]-direct[1])
                try:
                    print(path)
                except:
                    next
                                    
                cv2.line(map_image, (path[0].data.coords), (path[1].data.coords), (0, 255, 0), thickness=3, lineType=8)
                
                cv2.line(map_image, (path[1].data.coords), (path[2].data.coords), (0, 255, 0), thickness=3, lineType=8)
    plt.imshow(map_image),plt.show()

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
        for i in range(len(path)):
            if i<len(path)-1 and len(path)>1:
                coord = path[i].data.coords
                coord2 = path[i+1].data.coords
                cv2.line(map_image, (path[i].data.coords), (path[i+1].data.coords), (0, 255, 0), thickness=3, lineType=8)
                count+=1
    for arrow in arrows:
<<<<<<< Updated upstream:ArrowSearchTest.py
        if arrow.data.next_node == None:
            print('yes')

=======
        if arrow.data.next_node is not None:
            cv2.line(map_image, (arrow.data.coords), (arrow.data.next_node.data.coords), (0, 255, 0), thickness=3, lineType=8)
            
>>>>>>> Stashed changes:src/Pathmatching.py


    plt.imshow(map_image),plt.show()
    #tree_search1(arrows,tree)


main()

