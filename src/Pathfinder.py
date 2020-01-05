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

    goal_node = Node(None, end)

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

        maze[current_node.position[0]][current_node.position[1]] = 1



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
            # for closed_child in closed_list:
            #     if child == closed_child:
            #         continue

            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            
            print(child.h)



            ''' Modified Heuristic Algorithm
            1. find closest 5 arrows
            2. for each arrow, from closest to furthest:
                a. if the arrow points in roughly the same direction (the arrow points within 90 degrees on either side), this is the "next arrow"
                b. record its distance and break loop
            3. Check if the closest arrow is close enough
                a. if not, use normal heuristic
                b. if so, use modified heuristic:
                    i. h(n)' = (arrow_distance*Modifier + h(n)^2)*h(n)
                    this heuristic lowers as you get closer to the end, and is 0 when you are at the end
                    (i.e. shouldn't get trapped between arrow and endpoint)

            It is pulled to the correct arrows, but unfortunately it is difficult to raise the attraction of the arrows
            so that it goes in the correct lane, without causing it to fail to find a solution. 

            This is why it appears to be following the wrong arrows.



            '''
            ARROW_MODIF = 3
            MAX_DIST = 10000

            CLOSE_DIST = 2000
            arrow_dist = map_obj.root_arrow.search_knn((child.position[1],child.position[0]),5)

            direct2node = pt.direction(pt.Point(child.position[1],child.position[0]), pt.Point(end_node.position[1],end_node.position[0]))
            for arrow in arrow_dist:

                if pt.is_same_halfplane(direct2node, arrow[0].data.direction):
                    # print('route dir')
                    # print(direct2node)
                    # print('x coord')
                    # print(arrow[0].data.coords.x)
                    # print('y coord')
                    # print(arrow[0].data.coords.y)
                    # print('direction')
                    # print(arrow[0].data.direction)
                    next_arrow_dist = arrow[1]
                    break
                else:
                    next_arrow_dist = 0

            if next_arrow_dist < MAX_DIST:
                
                child.h = (next_arrow_dist*ARROW_MODIF + child.h)*child.h**0.5
                #child.k = (next_arrow_dist * child.h**0.5)*ARROW_MODIF
                #child.h = child.h



            #---------------------------------------------------------------

            goal_node.position = ()


            
            child.f = child.g + child.h



            # Child is already in the open list
            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue
            # for open_node in open_list:
            #     if child == open_node and child.g >= open_node.g:
            #         continue

            # Add the child to the open list
            open_list.append(child)


def main():
    #map_obj = MapHandler.MyMap(r"C:\Users\jmock\Documents\Projekt Arbeit Images\simpleTest.png")
    map_obj = MapHandler.MyMap()
    #maze.isObstacle(785,34)
    maze = map_obj.get_map()
    #x and y are flipped
    #start = (52,447)
    #end = (600,500)

    #map_obj.show_map()

    #map_obj.show_arrow_mask()


    points = map_obj.query_point()

    start = (points[0].y,points[0].x)
    end = (points[1].y,points[1].x)

    path = astar(map_obj, maze, start, end)
    map_obj.print_path(path)


if __name__ == '__main__':
    main()