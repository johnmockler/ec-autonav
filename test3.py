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
                                        if abs(delta2)<abs(path_cost) and abs(delta2)<=MAX_ANGLE:
                                            path = [root, first_node[0], second_node[0]]
                                            #path = [root, first_node[0]]
                                            #need path cost????
                                            direct = (first_dir, second_dir)
                            else:
                                second_dir = first_node[0].fwd_dir
                                delta2 = second_dir-first_dir
                                if abs(delta)<abs(path_cost) and abs(delta)<=MAX_ANGLE:
                                    path = [root, first_node[0], first_node[0].next_node]
                                    #path = [root, first_node[0]]
                                    direct = (first_dir, second_dir)
                            

                except:
                    # next  