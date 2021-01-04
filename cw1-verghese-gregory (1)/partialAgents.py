# partialAgents.py
# Verghese/28-Oct-2018
#
# Version 1
#
# The starting point for CW1.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Gregory Verghese, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util

'''
PartialAgent.py holds the logic for my agent solution to win the Pacman game.
The file contains two main classes. The first, the Search() class provides the code to
calculate an A* route from a start node to a goal Node with wall constraints as well.
This particular A* algorithm uses the Manhattan distance as it's heuristic, the function
for the Manhattan distance has been provided to me from the util.py file. Note I have also
included the BFS search function even though it is not implemented, as one of the strategies
I tested used a BFS. The second class, PartialAgent, takes in an agent object and uses the
api.py file to access informsation about its state and the world. This class then defines
three strategies which it exceutes with the following priority 1. Survival 2. Food and 3. Corners.
When a strategy returns None the next one is considered. Once a strategy passes back a direction this
is returned to the api.makeMove function and Pacman makes a move.
'''

class Search():
    '''
    A* search class,calculates the shortest route between two nodes in a map.
    Uses the Manhattan distance as the heuristic. The function has been provided
    from the util.py file.
    '''

    def __init__(self):

        self.nodesDict = {}

    def getNodeMeta(self, parentNode, childNode, goalNode):
        '''
        calculates g h and f heuristics for A* using number of Nodes from starts
        and manhattan distance for g and h respectively
        Args:
            parentNode:
            childNode:
            goalNode
        Returns:
            metaDict: Dictionary containing node info - heuristics and parent node.
        '''

        g = self.nodesDict[parentNode]['g'] + 1
        h = util.manhattanDistance(goalNode, childNode)
        f = g + h
        metaDict = {'parent': parentNode, 'f': f, 'g': g, 'h': h}

        return metaDict

    def backTrace(self, parentNode):
        '''
        given a end node from the A* seatch backtrace builds the A* shortest
        route using the parent information held in the costDict
        Args:
            parentNode: a tuple of (x, y) coordinates of the final node in the A* search
            shortest route
        Returns:
            route: a list of (x, y) tuples containing cartesian coordinates of the shortest
            route from start Node to end Node.
        '''

        route = []
        while parentNode is not None:
            route.append(parentNode)
            parentNode = self.nodesDict[parentNode]['parent']
        route.reverse()
        return route

    def getParentNode(self, openNodes):
        '''
        returns a node from the openNodes list based on the node with the
        lowest f value
        Args:
            openNodes: a list of tuples containing (x ,y) coordinates of nodes
            added to be searched but yet to be visited.
        Returns:
            parentNode: a tuple containing (x, y) cartesian coordinate of
            node with the lowest f value in openNodes
            '''

        parentNode = openNodes[0]
        nodeFs = [self.nodesDict[node]['f'] for node in openNodes]
        node = openNodes[nodeFs.index(min(nodeFs))]
        if self.nodesDict[node]['f'] < self.nodesDict[parentNode]['f']:
            parentNode = node

        return parentNode

    def getNeighbours(self, state, Node, walls):
        '''
        returns the North, East, South and West node neighbours to a given node using
        the mapping [(0, 1), (0, -1), (1, 0), (-1, 0)] for cartesian coordinates
        Args:
            state: object containing agent state information
            Node: tuple of (x, y) coordinates of node whose neighbours we are looking for
            walls: list of tuples containing (x, y) coordinates of walls in the map
        Returns:
            list of tuples containing (x, y) coordinates belonging to neighbours of the node
        '''

        Nodelst = [Node for i in range(4)]
        Mapping = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbours = list(map(lambda x, y: (x[0]+y[0], x[1]+y[1]), Mapping, Nodelst))

        return [n for n in neighbours if n not in walls]


    def Astar(self, state, startNode, goalNode, walls):
        '''
        given a starting Node on a graph and a goal node, along with contraints
        uses a A* search algorithm approach along with a Manhattan distance heuristic
        to return the shortest distance
        Args:
            state: object containing agent state information
            StartNode: (x, y) tuple containing cartesian coordinates of start node
            goalNode: (x, y) tuple containing cartesian coordinates of the goal node
            walls: list of tuples containing (x, y) coordinates of walls in the map
        Returns:
            self.bacTrace(parentNode) function which gives a list of the (x, y) coordinates along
            the shortest path.
        '''

        self.nodesDict[startNode] = {'parent': None, 'f': 0, 'g': 0, 'h': 0}
        openNodes = []
        visitedNodes = []
        openNodes.append(startNode)

        while openNodes:
            parentNode = self.getParentNode(openNodes)
            openNodes.pop(openNodes.index(parentNode))
            visitedNodes.append(parentNode)

            if parentNode == goalNode:
                return self.backTrace(parentNode)

            for childNode in self.getNeighbours(state, parentNode, walls):
                if childNode in visitedNodes:
                    continue

                self.nodesDict[childNode] = self.getNodeMeta(parentNode, childNode, goalNode)

                for node in openNodes:
                    if node == childNode and self.nodesDict[childNode]['g'] > self.nodesDict[node]['g']:
                        continue
                openNodes.append(childNode)



    def BFSearch(self, state, Start, goalNode, walls):
        '''
        NOTE THIS FUNCTION IS NOT USED
        given a starting Node on a graph and a goal node, along with contraints
        uses a BFS search algorithm approach to return the shortest distance
        Args:
            state: object containing agent state information
            StartNode: (x, y) tuple containing cartesian coordinates of start node
            goalNode: (x, y) tuple containing cartesian coordinates of the goal node
            walls: list of tuples containing (x, y) coordinates of walls in the map
        Returns:
            self.backTrace(parentNode) function which returns a list of the (x, y) coordinates along
            the shortest path.
        '''

        openNodes = util.Queue()
        visitedNodes = []

        openNodes.push([Start])
        test = util.Queue()
        test.push([Start])

        while not openNodes.isEmpty():

            path = openNodes.pop()
            rootNode = path[-1]

            print('start', Start)

            if rootNode not in visitedNodes:
                neighbours = self.getNeighbours(state, rootNode, walls)

            if rootNode not in visitedNodes:
                for node in neighbours:
                    nodeRoute = list(path)
                    nodeRoute.append(node)
                    openNodes.push(nodeRoute)

                    if node == goalNode:
                        print('nodeRoute', nodeRoute)
                        return nodeRoute

                visitedNodes.append(rootNode)


class PartialAgent(Agent):
    '''
    This class defines the strategies that my agent uses to play Pacman. There are
    three strategies, with the following order 1. Survival 2. Food 3. Corners. The getAction()
    function calls each one and passes back a direction to the api.makeMove() function. Each strategy has
    a getStategy() function that decides what needs to be done as well as a few helper functions.
    If a particular strategy is being initalized with a new goal the intializeRoute() function is called to set it up.
    If to Food and Corners strategy is a continuation from the previous move, then the next location on the A* route will
    be called. We use two generator functions, genMove() and genCorner() to store iterator objects in persistent storage,
    this allows for the internal state of the function to be remembered between successive calls and therfore A*
    route only has to be calculated once per goal.
    '''

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print "Starting up!"
        name = "Pacman"
        self.newGame = True
        self.strategy = None
        self.newFood = True
        self.newCorner = True
        self.walls = None
        self.corners = None
        self.ghstRad = 2
        self.survivalAstarRoute = None
        self.foodAstarRoute = None
        self.cornersAstarRoute = None
        self.legal = None
        self.pacWorld = True
        self.currentCorner = None
        self.nextCorner = self.genCorners()
        self.cornerMove = None
        self.foodMove = None
        self.last = Directions.STOP


    def initializeRoute(self, state, strategy, goal):
        '''
        initializes some of the class attribute for a chosen strategy
        Args:
            state: object containing agent state information
            strategy: a string containing the name of the strategy to be executed
            goal: (x, y) tuple containing cartesian coordinate of the goal Node
        '''

        if strategy == 'Survival':
            self.strategy = strategy
            #search = AstarRoute()
            #self.survivalAstarRoute = search.Astar(state, self.currLoc, goal, self.walls)

        elif strategy == 'Food':
            self.strategy = strategy
            search = Search()
            self.foodAstarRoute = search.Astar(state, self.currLoc, goal, self.walls)
            self.foodMove = self.genMove(self.foodAstarRoute)
            next(self.foodMove)
            self.newFood = False

        elif strategy == 'Corners':
            self.strategy = strategy
            search = Search()
            self.cornersAstarRoute = search.Astar(state, self.currLoc, goal, self.walls)
            self.cornerMove = self.genMove(self.cornersAstarRoute)
            next(self.cornerMove)

    def genCorners(self):
        '''
        generator function returns an iterator object to iterate through
        list of tuples containing (x, y) coordinates of corners. Remembers
        the index of the list reached between successive calls
        Returns:
            generator object which can be iterated over
        '''
        for i in range(len(self.corners)):
            yield(self.corners[i])

    def genMove(self, route):
        '''
        generator function returns an iterator object to iterate through
        list of tuples containing (x, y) coordinates of a route.
        Remembers the index of the list reached between successive calls
        Returns:
            generator object which can be iterated over
        '''
        for i in range(len(route)):
            yield(route[i])

    def getAction(self, state):
        ''' This begins the decision making process for the strategy employed
        each move in the game. Sets up some of the state related class attributes by calling
        the api, begins building getWorldMap and sets newGame to False. Controls the flow and
        priority of the three game strategies availiable, there priority ordering is as follows
        1.survival 2. food 3. corners.
        Returns
            api.makeMove(xxxxx, self.legal) where xxxxx is the direction of travel Pacman needs to make
            for the strategy just employed.
            '''

        if self.newGame == True:
            self.corners = api.corners(state)
            self.walls = api.walls(state)
            self.getWorldMap(state)
            self.newGame = False

        self.currLoc = api.whereAmI(state)
        self.legal = api.legalActions(state)

        survival = self.getSurvival(state)
        if survival != None:
            print('pacman is using the survival strategy', survival)
            return api.makeMove(survival, self.legal)

        food = self.getFood(state)
        if food != None:
            print('pacman is using the food strategy', food)
            return api.makeMove(food, self.legal)

        corners = self.getCorners(state)
        if corners != None:
            print('pacman is using the corner strategy', corners)
            return api.makeMove(corners, self.legal)



    def getSurvival(self, state):
        '''gets the survival strategy and returns an escape direction
        if ghosts are in the safe area.
        Args:
            state: object containing agent state information
            returns: api.makeMove(random.choice(self.legal), self.legal)
        '''
        #locate the ghosts and see if they exist in our safe area
        ghosts = api.ghosts(state)
        ghostLoc = self.checkGhosts(ghosts)

        if len(ghostLoc) == 0:
            return None
        #Find a set of moves that would help us escape
        escape = self.planEscape(state, ghostLoc)

        if len(escape) == 0:
            return None

        for direc in escape:
            if direc in self.legal:
                self.legal.remove(direc)

        if len(self.legal) > 1:
            self.legal.remove(Directions.STOP)

        return api.makeMove(random.choice(self.legal), self.legal)


    def planEscape(self, state, ghostLoc):
        '''
        Takes ghost locations and for each ghost initializes survival strategy
        which returns an escape strategy. This is stored in the list escape.
        Args:
            state: object containing agent state information
            ghostLoc: list of (x, y) tuples containing ghost locations
        Returns:
            escape: list of escape directions, e.g 'North'

        '''
        escape = []
        #loop through each ghost and call the getQuadMapping find the dangerous directions
        for ghost in ghostLoc:
            self.initializeRoute(state, 'Survival', ghost)
            directions = self.getQuadMapping(ghost)
            escape.extend(directions)
        return escape


    def getQuadMapping(self, ghost):
        '''
        Split the map up into four quadrants and two axis. Find where the ghosts
        are and remove the corresponding directions. If they are in a quadrant we
        remove two directions. E.g if it's the upper right quadrant we remove
        North and East. If they are on a axis remove just the one.
        Args:
            ghost: a (x, y) tuple containing ghost location
        Returns:
            a list of directions that we want to remove from our set of possible legal moves
        '''
        if (ghost[0] - self.currLoc[0]) < 0 and (ghost[1] - self.currLoc[1]) < 0:
            return ['West', 'South']
        elif (ghost[0] - self.currLoc[0]) < 0 and (ghost[1] - self.currLoc[1]) > 0:
            return ['West', 'North']
        elif (ghost[0] - self.currLoc[0]) > 0 and (ghost[1] - self.currLoc[1]) > 0:
            return ['East', 'North']
        elif (ghost[0] - self.currLoc[0]) > 0 and (ghost[1] - self.currLoc[1]) < 0:
            return ['East', 'South']
        elif (ghost[0] - self.currLoc[0]) == 0 and (ghost[1] - self.currLoc[1]) < 0:
            return ['South']
        elif (ghost[0] - self.currLoc[0]) == 0 and (ghost[1] - self.currLoc[1]) > 0:
            return ['North']
        elif (ghost[0] - self.currLoc[0]) < 0 and (ghost[1] - self.currLoc[1]) == 0:
            return ['West']
        elif (ghost[0] - self.currLoc[0]) > 0 and (ghost[1] - self.currLoc[1]) == 0:
            return ['East']

    def getSafeArea(self, currLoc):
        '''
        works out a safe area around Pacman based on the radius self.ghstrad and returns the coordinates
        Args:
            currLoc:
        Returns:
            safeArea: returns a list of (x, y) tuples mapping the grid around Pacman
        '''

        mapping = [(x,y) for x in range(self.ghstRad + 1) for y in range(self.ghstRad + 1)]
        quad = lambda z: ((z[0]*1, z[1] * 1), (z[0]*(-1), z[1] * 1), (z[0]*(1), z[1] * (-1)), (z[0]*-1, z[1] * -1))
        mapping = list(set([n for loc in map(quad, mapping) for n in loc]))
        currLocLst = [currLoc for i in range(len(mapping))]
        safeArea = list(map(lambda x, y: (max(x[0]+y[0],0), max(x[1]+y[1],0)), mapping, currLocLst))

        return safeArea

    def checkGhosts(self, ghosts):
        '''
        checks to see if any of the ghosts are within the safeArea
        Args:
            ghosts: a list (x, y) tuples that give locations of ghosts
        Returns:
            [n for n in ghosts if n in safeArea] a list of ghosts within the safeArea
        '''
        safeArea = self.getSafeArea(self.currLoc)
        return [n for n in ghosts if n in safeArea]


    def getFood(self, state):
        '''
        gets the food strategy and returns an escape direction
        if ghosts are in the safe area.
        Args:
            state: object containing agent state information
        Returns:
            api.makeMove(direction, self.legal)
        '''
        # Add capsules to the food
        foodLocations = api.food(state)
        capsules = api.capsules(state)
        foodLocations.extend(capsules)

        self.mapWorld(self.currLoc, foodLocations)

        if len(foodLocations) == 0:
            foodLocations = [node for node in self.pacWorld if self.pacWorld[node] == 'f']
        # can't find any food so return None and use corner strategy
        if len(foodLocations) == 0:
            return None

        direction = self.getFoodAction(foodLocations, state)

        return api.makeMove(direction, self.legal)

    def getFoodAction(self, foodLocations, state):
        '''
        gets our next action in the food stategy. Either continue
        traversing the current A* route or intialize a new one with a new
        food goal.
        Args:
            foodLocations: a list of tuples of (x, y) food locations.
            state: object containing agent state information
        Returns:
            self.chaseFood()
        '''
        if self.newFood == False and self.strategy == 'Food':
            return self.chaseFood()

        closestFood = self.getClosestFood(foodLocations)
        self.initializeRoute(state, 'Food', closestFood)

        return self.chaseFood()


    def chaseFood(self):
        '''
        gets the next node on the route from the generator object self.foodMove
        and returns correspnding direction.
        Returns:
                Direction: returns string of direction to move. e.g 'North'
        '''
        #call the generator function stored in self.foodMove to get next location in route
        nextNode = next(self.foodMove)
        if nextNode == self.foodAstarRoute[-1]:
            self.newFood = True
        Direction = self.mapDirection(self.currLoc, nextNode)
        return Direction

    def mapDirection(self, currNode, nextNode):
        '''
        maps the direction to take to get from current node to the next node. uses a static
        dictionary with (x, y) coordinates, e.g (0, 1) for 'North'.
        Args:
            currNode: (x, y) tuple with coordiantes of current node
            nextNode: (x, y) tuple with coordiantes of next node
        Returns:
            String with the correct direction, e.g 'North'

        '''
        direcDict = {(0, 1): 'North', (0, -1): 'South',(-1, 0): 'West', (1, 0): 'East'}
        return [direcDict[k] for k in direcDict if (k[0] + currNode[0], k[1] + currNode[1]) == nextNode][0]

    def getClosestFood(self, foodLoc):
        '''
        gets distances to the food that visble and returns food that is
        closest
        Args:
            foodloc: list of (x, y) tuples containg food locations
        Return:
            (x, y) food node with the minimum distance

        '''
        foodDistances = self.getDistances(self.currLoc, foodLoc)
        return foodLoc[foodDistances.index(min(foodDistances))]

    def getDistances(self, currLoc, goalLocs):
        '''
        uses the manhattanDistance provided with the util.py file to return a list
        of distances to each food node that is visible
        Args:
            currLoc: currNode: (x, y) tuple with coordinates of current node
            goalLocs: list of (x, y) tuples with coordinates of target nodes.
        '''
        return [util.manhattanDistance(currLoc, loc) for loc in goalLocs]

    def getWorldMap(self, state):
        '''builds a map of the pacman world. Finds the boundaries of the grid
        and then builds a map out of list of (x, y) tuples by iterating to each boundary.
        Stores the locations along with a status in the dictioary self.pacWorld. Status is
        initially set to be 'o' for open or 'w' for wall.
        Args:
            state: object containing agent state information
        '''

        boundaries = [loc for loc in api.corners(state) if loc[0] and loc[1] != 0][0]
        map = [(x,y) for y in range(boundaries[1]) for x in range(boundaries[0])]
        status = ['o' for i in range(len(map))]

        #create the world by setting all open locations to 'o' walls to 'w'
        self.pacWorld = dict(zip((map), (status)))
        for k in self.pacWorld:
            if k in self.walls:
                self.pacWorld[k] = 'w'

    def mapWorld(self, node, foodLoc):
        '''
        We take the current location and any food locations we can see and update the pacWorld
        Dictionary to 'v' and 'f' respectively
        Args:
            node: (x, y) tuple of current location
            foodLoc: list of (x, y) tuples containing locations of food.
        '''
        for food in foodLoc:
            self.pacWorld[food] = 'f'
        self.pacWorld[node] = 'v'


    def getCorners(self, state):
        '''
        this gets the corner strategy move. First we determine the corners from
        the boundaries, each corner is coordinate is +1 if it is 0 or -1 otherwise.
        checks the corners have not all been revisited and calls reset otherwise.
        getCornerAction is then called to return the direction for the next move.
        Args:
            state: object containing agent state information
        Returns:
            api.makeMove with the correspondong direction that needs to be made
            for the next location in the corner route.
        '''
        boundaries = [[(loc[i] - 1) if loc[i] != 0 else (loc[i] + 1)
                        for loc in api.corners(state)] for i in range(2)]

        self.corners = zip(boundaries[0],boundaries[1])

        #check to see if all the corners have been visited
        if 'o' not in [self.pacWorld[c] for c in self.corners]:
            self.resetCorners()

        self.mapWorld(self.currLoc, api.food(state))

        Direction = self.getCornerAction(state)
        return api.makeMove(Direction, self.legal)


    def getCornerAction(self, state):
        '''
        Determine which corner action should be returned. If newCorner == True then it
        initializes a new route to the next corner. if we want to stick with the current corners
        (newCorner == False) and the last strategy was not corner then it initializes a new route for
        the current corner. Else if we are still on the current corner and the last strategy was corner
        we continue on the same route.
        Args:
            state: object containing agent state information
        Returns:
            self.chaseCorner() which returns a the direction for the next move
        '''


        if self.newCorner == False and self.strategy == 'Corners':
            print('not new')
            return self.chaseCorner()

        elif self.newCorner == False and self.strategy != 'Corners':
            self.initializeRoute(self.currLoc, 'Corners', self.cornersAstarRoute[-1])

            # if we have landed on this corner via a different strategy then we just stop and set self.newCorner to true
            if self.currLoc == self.cornersAstarRoute[-1]:
                self.newCorner = True
                return 'Stop'
            else:
                return self.chaseCorner()

        elif self.newCorner == True:
            print('new everything')
            self.initializeRoute(self.currLoc, 'Corners', next(self.nextCorner))
            self.newCorner = False
            return self.chaseCorner()


    def chaseCorner(self):
        '''
        calls the next generator function genMove stored in self.cornerMove to iterate
        to the next corner. Maps the coresponding direction for pacman to move to nextNode.
        If the next node is the same as the corner then we set self.newCorner to True.
        Returns:
            Direction: String of the direction pacman needs to take to make his move
            from current location to the next location, e.g 'North'
        '''


        nextNode = next(self.cornerMove)
        if nextNode == self.cornersAstarRoute[-1]:
            self.newCorner = True
        Direction = self.mapDirection(self.currLoc, nextNode)
        return Direction

    def resetCorners(self):
        '''
        resets corners by instantiating self.genCorners and storing in self.nextCorner.
        Sets newCorner ==True and updatesin self.pacWorld to show the corners as open
        '''
        print('RESETTING the corners to open', [self.pacWorld[c] for c in self.corners])
        self.nextCorner = None
        self.nextCorner = self.genCorners()
        self.newCorner = True
        self.pacWorld.update({c:'o' for c in self.corners})

    def randomishAgent(self, state):
        '''NOTE THIS FUNCTION IS NOT USED
        Makes a random selection and keeps doing it until it can't
        Args:
            State: object containg agent information
        '''
        # Get the actions we can try, and remove "STOP" if that is one of them.
        self.legal = api.legalActions(state)
        if Directions.STOP in self.legal:
            self.legal.remove(Directions.STOP)
        # If we can repeat the last action, do it. Otherwise make a
        # random choice.
        if self.last in self.legal:
            return self.last
        else:
            pick = random.choice(self.legal)
            # Since we changed action, record what we did
            self.last = pick
            return pick


    # This is what gets run in between multiple games
    def final(self, state):
        '''
        resets class attributes between instances of games to intial state
        '''
        print "This game has finished now. Thanks!"
        self.newGame = True
        self.strategy = None
        self.newFood = True
        self.newCorner = True
        self.walls = None
        self.corners = None
        self.survivalAstarRoute = None
        self.foodAstarRoute = None
        self.cornersAstarRoute = None
        self.legal = None
        self.pacWorld = None
        self.currentCorner = None
        self.nextCorner = self.genCorners()
        self.cornerMove = None
        self.foodMove = None
