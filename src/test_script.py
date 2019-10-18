import MyMap

def main():


    maze = MyMap.MyMap()
    maze.loadMap()
    arrows = maze.findArrows()

    print(len(arrows))




if __name__ == '__main__':
    main()