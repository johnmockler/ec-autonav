import cv2
import numpy as np
from matplotlib import pyplot as plt
import math as m
from include import kdtree
from src import Arrow
import random

MAP = '/home/john/Pictures/ENC_test.png'

#MAP = '/home/john/Pictures/simpleTest.png'
TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])

#process map
map_image = cv2.imread(MAP,1)
hsv_img = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_threshed = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)


def detectArrows(mask):
    #pre-process map before input to function  
    corners = cv2.goodFeaturesToTrack(mask,500,0.5,10)
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

def tree_search(node,tree,direct=None,layer=1):
    MAX_ANGLE = m.pi/8.0
    MAX_LAYER = 3
    best_path = []
    best_cost = 1000
    best_dir = 0
    optimal_path = []

    #run initial code
    #create path here????

    ##DONT FORGET TO ADD NODE TO PATH!!!!
    if layer==1:
        if node.data.prev_node is not None:
            direct = node.data.bwd_dir
            layer+=1
            optimal_path = [node.data.prev_node, node]
        else:
            optimal_path = [node]
        if node.data.next_node != None:
            return best_path,best_cost

    next_layer = tree.search_knn(node.data.coords,5)

    if layer < MAX_LAYER:
        for branch_node in next_layer:
            temp_cost = 0
            #use node here or branch_node????
            temp_path = [branch_node]

            if branch_node[1]!=0 and branch_node[0].data.prev_node==None:

                branch_direct = direction(node.data.coords,branch_node[0].data.coords)

                if direct is not None:
                    delta = branch_direct
                    layer+=1
                else:
                    delta = None
                    layer+=1

                if (delta!=None and abs(delta)<=MAX_ANGLE) or delta == None:
                    if branch_node[0].data.next_node==None:
                        fwd_cost,fwd_path = tree_search(branch_node[0],tree,branch_direct,layer)
                    else:
                        layer+=1
                        fwd_delta = branch_node[0].data.fwd_dir - branch_direct
                        fwd_cost,fwd_path = tree_search(branch_node[0].data.next_node,tree,branch_node[0].data.fwd_dir,layer)
                        fwd_cost = m.mean(abs(fwd_delta),fwd_cost)
                        fwd_path = [branch_node[0].data.next_node] + fwd_path

                    if delta == None:
                        temp_cost = fwd_cost
                        temp_path = temp_path.append(fwd_path)
                    else:
                        temp_cost = m.mean(fwd_cost,abs(delta))
                        temp_path = temp_path.append(fwd_path)
                
                    if temp_cost < best_cost:
                        best_cost = temp_cost
                        best_path = temp_path
                        best_dir = branch_direct


        #dont forget to connect the first node with optimal..
        for i in range(len(best_path)):
            if i<len(best_path):
                #find way to save direction...
                best_path[i].data.next_node = best_path[i+1]
                best_path[i].data.fwd_dir = best_dir
                best_path[i+1].data.prev_node = best_path[i]
                best_path[i+1].data.bwd_dir = best_dir


        #CONSIDER THIS ALGORITHM!!!
        if len(optimal_path)==1:
            optimal_path[0].data.next_node = best_path[0]
            optimal_path[0].data.fwd_dir = 0

        return best_cost, best_path

    else:
        for branch_node in next_layer:
            #consider allowing better nodes to replace...
            if branch_node[1]!=0 and branch_node[0].data.prev_node==None:
                branch_direct = direction(node.data.coords,branch_node[0].data.coords)
                delta = branch_direct - direct
                temp_path = [branch_node[0]]
                temp_cost = abs(delta)
                if temp_cost<best_cost and temp_cost<=MAX_ANGLE:
                    best_path = temp_path
                    best_cost = temp_cost
                    best_dir = branch_direct
        

        #dont forget to update the node objects!!!!
        for i in range(len(best_path)):
            if i<len(best_path):
                best_path[i].data.next_node = best_path[i+1]
                best_path[i].data.fwd_dir = best_dir
                best_path[i+1].data.prev_node = best_path[i]
                best_path[i+1].data.bwd_dir = best_dir

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

# def search_tree(node,tree,path=[],cost=-1,direct,depth=3):
#     MAX_ANGLE = m.pi/8.0

#     if node.data.next_node is not None:
#         return best_path,best_cost

#     while len(best_path)<depth:
#         if node.data.prev_node is not None:
#             direct = 


#         next_layer = tree.search_knn(node.data.coords,5)

#     return best_path,best_cost





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
    plt.imshow(map_image),plt.show()
    #tree_search1(arrows,tree)


main()

    


# #recursive search function
# def search_tree(node0,node1,tree, max_depth):
#     depth = 0;
#     if node1.data.prev_node ==None and node1[1]!=0:
#         direct = direction(node0.data.coords, node1.data.coords)



    # MAX_ANGLE = m.pi/4.0
    # for arrow in arrows:
    #     if arrow.data.next_node is None:
    #         first_set = arrow_tree.search_knn(arrow.data.coords,6)
    #         branch = []
    #         for first_leaf in first_set:
    #             if arrow.data.prev_node!=None:
    #                 #calculate angle between the nodes 
    #                 angle1 = arrow.data.bwd_dir()
    #                 print(angle1)
    #                 angle2 = direction(arrow.data.coords, first_leaf[0].data.coords)
    #                 delta1 = angle1 - angle2
    #                 if abs(delta1 < MAX_ANGLE):
                
    #                     dist = first_leaf[1]
    #                     #print(arrow.data.coords)
    #                     #print(first_leaf[0].data.coords)
    #                     theta1 = direction(arrow.data.coords, first_leaf[0].data.coords)
    #                     #print(theta1)
    #                     second_set = arrow_tree.search_knn(first_leaf[0].data.coords,7)
    #                     delta = 1000
    #                     next_node = None
    #                     for second_leaf in second_set:
    #                         #dont search the same or the original node
    #                         if second_leaf[1]!=0 and second_leaf[0].data.coords!=arrow.data.coords:
    #                             theta2 = direction(first_leaf[0].data.coords, second_leaf[0].data.coords)
    #                             leaf_delta = theta2 - theta1
    #                             if (delta == None and (abs(delta) > MAX_ANGLE )) or abs(delta)> abs(leaf_delta):
    #                                 delta = leaf_delta
    #                                 next_node = second_leaf[0]
    #                     branch.append((first_leaf[0],next_node,delta))
    #             elif first_leaf[1]!=0 and first_leaf[0].data.prev_node==None:
    #                 dist = first_leaf[1]
    #                 #print(arrow.data.coords)
    #                 #print(first_leaf[0].data.coords)
    #                 theta1 = direction(arrow.data.coords, first_leaf[0].data.coords)
    #                 #print(theta1)
    #                 second_set = arrow_tree.search_knn(first_leaf[0].data.coords,7)
    #                 delta = 1000
    #                 next_node = None
    #                 for second_leaf in second_set:
    #                     #dont search the same or the original node
    #                     if second_leaf[1]!=0 and second_leaf[0].data.coords!=arrow.data.coords:
    #                         theta2 = direction(first_leaf[0].data.coords, second_leaf[0].data.coords)
    #                         leaf_delta = theta2 - theta1
    #                         if (delta == None and (abs(delta) > MAX_ANGLE )) or abs(delta)> abs(leaf_delta):
    #                             delta = leaf_delta
    #                             next_node = second_leaf[0]
    #                 branch.append((first_leaf[0],next_node,delta))
    #     d_min = None
    #     point1 = None
    #     point2 = None
    #     for node in branch:
    #         #print(leaf[2])
    #         #print(d_min)
    #         # if  node[0].data.next_node!=None:
    #         #     angle2 = node[0].data.fwd_dir()
    #         #     angle1 = direction(arrow.data.coords, node[0].data.coords)
    #         #     if abs(angle2-angle1)<MAX_ANGLE and (d_min == None or abs(node[2]) < abs(d_min)) :
    #         #         d_min = node[2]
    #         #         point1 = node[0]
    #         #         point2 = node[1]
    #         if (d_min == None or abs(node[2]) < abs(d_min)):
    #             d_min = node[2]
    #             point1 = node[0]
    #             point2 = node[1]

        # arrow.data.next_node = point1
        # #point1.data.next_node = point2
        # #cv2.circle(map_image, arrow.data.coords,3,255,-1)
        # #cv2.circle(map_image, point1.data.coords,3,255,-1)
        # #cv2.circle(map_image,point2.data.coords,3,255,-1)
        # cv2.line(map_image, (arrow.data.coords), (point1.data.coords), (0, 255, 0), thickness=3, lineType=8)
        # #cv2.line(map_image, (point1.data.coords), (point2.data.coords), (0, 255, 0), thickness=3, lineType=8)

        # plt.imshow(map_image),plt.show()






#detectArrows(map_mask)