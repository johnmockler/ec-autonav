tree_search2(node, tree, ptA = None count = 1):
''' this algorithm minimizes the perpendicular distance of the items in the tree, rather than minimizing the angle.
also streamlines the code.
'''
    CYCLES = 3
    SEARCH_LIMIT = 4500.0
    MAX_DISTANCE = 20.0

    if count == 1:
        if node.data.next_node is not None:
                return 0
            else:
                pre_path = [node]
                if node.data.prev_node is not None:
                    ptA = node.data.prev_node.data.coords
                    ptB = node.data.coords
                    count += 1
                else:
                    ptA = node.data.coords
                    ptB = None
        
    next_layer = tree.search_knn(node.data.coords, 5)
    next_layer = list(filter(lambda x: x[1] == 0 or x[1] > SEARCH_LIMIT, next_layer))

    if count < CYCLES:
        for branch_node in next_layer:
            #we need to collect at least two points first. 

            if count == 0:
                ptB = branch_node[0].data.coords
                node = branch_node[0]
                path.append(node)
                count += 1

            else:
                ptP = branch_node[0].data.coords
                branch_cost = perpendicularDist(ptA, ptB, ptP)

                if branch_cost < MAX_DISTANCE:
                    if branch_node[0].data.next_node == None:
                        fwd_cost, fwd_path = tree_search2(branch_node[0],tree,ptB)


