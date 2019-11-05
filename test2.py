def tree_search(node, tree, depth, first_loop=True, path=[],  cost=None, direct=None):
    MAX_ANGLE = m.pi/8.0
    best_path = []
    best_cost = 1000
    curr_path = path
    if cost is not None:
        curr_cost = cost
    else:
        curr_cost = 1000

    if first_loop:
        #for some reason i will get an error if I put this directly into the if statement
        prevnode_test = node.data.prev_node is not None
        #Only for first node: if it is already part of a tree, check the direction
        if prevnode_test:
            bwd_direct = node.data.bwd_dir
            curr_path = [node.data.prev_node, node]
            depth-=1
        else:
            bwd_direct = None
            curr_path.append(node)
            
        if node.data.next_node is not None:
            return best_path, best_cost
    else:
        bwd_direct = None

    if depth > 0:
        next_layer = tree.search_knn(node.data.coords,5)
        
        for branch_node in next_layer:
            if branch_node[1]!=0 and branch_node[0].data.prev_node==None:
                
                branch_direct = direction(node.data.coords,branch_node[0].data.coords)
                curr_path.append(branch_node[0])

                bwddir_test = bwd_direct is not None
                if bwddir_test:
                    delta = abs(branch_direct-bwd_direct)
                    curr_cost+=delta
                    depth -= 2
                elif direct is not None:
                        delta= abs(branch_direct-direct)
                        curr_cost+=delta
                        depth -= 1

                else:
                    delta = None
                    curr_cost = 0;
                    depth -= 1

                #print(curr_cost)
                print(len(curr_path))

                if (delta!=None and abs(delta)<=MAX_ANGLE) or delta==None:                       
                    
                    if branch_node[0].data.next_node==None:
                        curr_path, curr_cost = tree_search(branch_node[0],tree,depth,False,curr_path,curr_cost,branch_direct)
                    else:
                        depth-= 1
                        delta1 = branch_node[0].data.fwd_dir - branch_direct
                        curr_cost+=delta1
                        curr_path.append(branch_node[0].data.next_node)

                        if depth > 0:
                            curr_path, curr_cost = tree_search(branch_node[0].data.next_node,tree,depth,False, curr_path,curr_cost,branch_node[0].data.fwd_dir)

                if curr_cost < best_cost:
                    best_path = curr_path
                    best_cost = curr_cost
    else:
        best_path = curr_path
        best_cost = curr_cost

    return best_path, best_cost