#https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2

import numpy as np
import MapHandler
import PointHandler as pt
from include import kdtree
import math as m

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.k = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(map_obj, maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            #if map_obj.is_obstacle(node_position[0], node_position[1]) != 0:
            if maze[node_position[0],node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            
            print(child.h)
            #configure cost function k 
            ARROW_MODIF = .03
            DIST_MODIF = 10
            MAX_DIST = 10000
            arrow_dist = map_obj.root_arrow.search_knn((child.position[1],child.position[0]),5)
            modified_dist = []

            #for arrow in arrow_dist:
            #    dist = pt.distance(arrow[0].data.coords, pt.Point(child.position[1],child.position[0]))
            #    modified_dist.append(dist**2*arrow[1]/16.0)


            # don't mess with this algorithm. it works and I don't know why...
            direct2node = pt.direction(pt.Point(child.position[1],child.position[0]), pt.Point(end_node.position[1],end_node.position[0]))
            for arrow in arrow_dist:


                if pt.is_same_halfplane(direct2node, arrow[0].data.direction):
                    print('arrow coord')
                    print(arrow[0].data.coords.x)
                    print('arrow dir')
                    print(arrow[0].data.direction)
                    print('direction2end')
                    print(direct2node)
                    next_arrow_dist = arrow[1]
                    break
                else:
                    next_arrow_dist = 0


            if next_arrow_dist < MAX_DIST:
                child.k = (next_arrow_dist * child.h**0.5)*ARROW_MODIF
            else:
                child.k = 0

            
            child.f = child.g + child.h + child.k



            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


def main():
    map_obj = MapHandler.MyMap()
    #maze.isObstacle(785,34)
    maze = map_obj.get_map()
    #x and y are flipped
    #start = (52,447)
    #end = (600,500)


    points = map_obj.query_point()

    start = (points[0].y,points[0].x)
    end = (points[1].y,points[1].x)
    path = astar(map_obj, maze, start, end)
    map_obj.print_path(path)


if __name__ == '__main__':
    main()