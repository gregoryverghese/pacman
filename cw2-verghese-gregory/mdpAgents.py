
# mdpAgents.py
# verghese/08-Dec-2018
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# https://emea01.safelinks.protection.outlook.com/?url=http%3A%2F%2Fai.berkeley.edu%2F&amp;data=01%7C01%7Cgregory.verghese%40kcl.ac.uk%7C13ed618722834c562ae408d659d4040e%7C8370cf1416f34c16b83c724071654356%7C0&amp;sdata=zfxK0FC9W1jAJP2hZ%2FjvQivpjkWJo1uOUiktbYQiynA%3D&amp;reserved=0
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to https://emea01.safelinks.protection.outlook.com/?url=http%3A%2F%2Fai.berkeley.edu&amp;data=01%7C01%7Cgregory.verghese%40kcl.ac.uk%7C13ed618722834c562ae408d659d4040e%7C8370cf1416f34c16b83c724071654356%7C0&amp;sdata=fpH%2Bqnoaw5XDLzgxF4w61F8W6IIdk%2FlDfx02DNytpqM%3D&amp;reserved=0.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util

import time

'''
MDPSolver class contains logic for calculating the utilities of a set of states
using a value iteration algorithm. Class attributes are:

1. states, s in S, a set of avaliable states in the environment
2. currState, s, the state pacman is in
3. actions, a in A, a set of actions where action, a, takes the agent from state, s, to state s'.
4. transitionModel, P(s'|s,a), a transiton model, that describes the probability of each action
5. RewardFunction, R(s), determines rewards an agent receives for being in certain states
6  gamma, decay factor in the bellman equation
7. conver, convergence tolerance in value iteration.

A value iteration method loops through each state and  calls a bellman equation function
which in turn gets the maximum expected utility for each state based on it's adjacent neighbours.

'''
class MDPSolver():

    def __init__(self, currState, states):
        self.states = states
        self.currState = currState
        self.actions = ['North', 'South', 'West', 'East']
        self.transitionModel = {'Sucess': 0.8, 'Failure': 0.1}
        self.RewardFunction = self.setRewards({'o': -1, 'f':10, 'g':-500})
        self.gamma = 0.6
        self.conver = 0.1

    '''
    Loops through self.states and applies the MDP reward function based on the status
    of each state. Ignores walls.
    open state, 'o': -1
    food state, 'f': 10
    open state, 'g': -500
    Args:
        @rewards: dictonary containing reward function values
    '''
    def setRewards(self, rewards):
        for s in self.states:
            if s.status != 'gz':
                s.reward = rewards[s.status]


    '''
    Loops through self.states and intializes each state to 0.
    '''
    def setUtilities(self):
        for s in self.states:
                s.utility = 0


    '''
    Begins the value iteration process to determnine utilities of every state.
    Number of iterations determined by convergence factor self.conver. Calls
    the bellmanUpdate to get the utilties of each state.
    '''
    def valueIteration(self):

        self.setUtilities()

        check = [False]
        while False in check:
            utilities = [(s, self.bellmanUpdate(s)) for s in self.states]
            check = self.getConvergence(utilities)

            x = [(u[0].getLocation(), u[1]) for u in utilities]

            for s in utilities:
                s[0].utility = s[1]


    '''
    checks the Convergence of each state utility against previous iteration
    Args:
        @utilities: list of utility values
    Returns:
        hasConverged: list of boolean values indicating convergence for each state.
    '''
    def getConvergence(self, utilities):

        convergence = [abs(s[0].utility - s[1]) for s in utilities]
        hasConverged = [True if c < self.conver else False for c in convergence]

        return hasConverged



    '''
    Calculates the bellman equations to get utiltiy of a state.
    For each state the utility is calculated using the maximum expected
    utiltiy, the reward and the MDP gamma.
    Args:
        @state: node object containing each states status, reward and utility
        information
    Returns:
        utility: float value containing the utility for the state.
    '''
    def bellmanUpdate(self, state):

        meu = self.getMEU(state, self.actions)[1]
        currLoc = self.currState.getLocation()
        stateLoc = state.getLocation()
        decay = abs(currLoc[0] - stateLoc[0]) + abs(currLoc[1] - stateLoc[1])
        utility = state.reward + (self.gamma * meu)
        return utility

    '''
    Calculates the maximum expected utility for each state, based on self.transitionModel
    the states adjacent state utilities.
    Args:
        @state: node object containing each states status, reward and utility
        information
        @actions: gamestate string direction. Either 'North', 'South', 'East', 'West'
    Returns:
        meu: tuple, (state, meu), containing node object and its corresponding utility.

    '''
    def getMEU(self, state, actions):

        expUtildict = {}

        for move in actions:
            dirVec = self.vector(move)
            targetLoc = tuple(map(sum, zip(state.getLocation(), dirVec)))
            targetState = next((n for n in self.states if n.getLocation() == targetLoc), None)
            targetState = state if targetState == None else targetState


            adjStates = self.getAdjState(dirVec, state)
            moveExpUtil = self.getExpUtility(adjStates, targetState)

            expUtildict[move] = moveExpUtil

        meu = max(expUtildict.values())

        return next(((k, v) for k, v in expUtildict.items() if v == meu), None)

    '''
    returns vectorized representation of directions.
    Args:
        @move: string with move direction. North, South, East or West
    Returns:
        tuple containg vector representing direction of move.
    '''
    def vector(self, move):

        direcDict = {'North': (0, 1), 'South': (0, -1), 'West': (-1, 0), 'East': (1, 0)}
        return  direcDict[move]


    '''
    gets the expected utility of each the state based on the transitionModel, the
    utiltiy of the target state and adjacent states.
    Args:
        @adjStates: list of adjacent state objects
        @targetState: the target state object
    Returns:
        expUtility: float of the expected utility for the state of interest.
    '''
    def getExpUtility(self, adjStates, targetState):


        adjUtil = sum([self.transitionModel['Failure'] * s.utility for s in adjStates])
        targetStateUtil = self.transitionModel['Sucess'] * targetState.utility
        expUtility = adjUtil + targetStateUtil

        return expUtility


    '''
    Determines adjacent states to the targetState
    Args:
        @dirVec: tuple containg direction vector
        @state: state object
    Returns:
        adjStates: list of adjacent states
    '''
    def getAdjState(self, dirVec, state):

        adjMapping =[(1, 0), (-1, 0)] if dirVec[0] == 0 else [(1, 0), (-1, 0)]
        adjLoc = [tuple(map(sum, zip(d, state.getLocation()))) for d in adjMapping]
        adjStates = [s for loc in adjLoc for s in self.states if s.getLocation() == loc]

        for i in range(2-len(adjStates)):
            adjStates.append(state)

        return adjStates

'''
This class defines our MDPAgent. Calls PacWorld class to map environment
and then passes node information as states to MDPSolver and calls the value
iteration algorithm. Gets the policy from MDP solver and passes the policy move
to the api.
'''
class MDPAgent(Agent):

    def __init__(self):
        #print "Starting up MDPAgent!"
        name = "Pacman"
        self.newGame = None
        self.walls = None
        self.world = None
        self.ghostRadius = 2
        self.startT = None


    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):

        self.startT = time.time()
        #print "Running registerInitialState for MDPAgent!"
        #print "I'm at:"
        #print api.whereAmI(state)


    # This is what gets run in between multiple games
    def final(self, state):
        self.startT = None



    '''
    Calls PacWorld and MDPSolver and gets pacmans next move
    based on maximum expected utility, meu, and passes it to
    the api.
    Args:
        @state: gamestate
    Returns:
        api.makeMove(meuMove, api.legalActions(state))
    '''
    def getAction(self, state):

        currLoc = api.whereAmI(state)
        self.world = PacWorld(state)
        legalActions = api.legalActions(state)

        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        self.buildGhosts(self.world.nodes)

        mdp = self.getMDP(currLoc)
        meuMove = mdp.getMEU(mdp.currState, legalActions)[0]

        return api.makeMove(meuMove, api.legalActions(state))

        '''
        Builds our MDP object and calls valueIteration
        Args:
            @currLoc: tuple of coordinates with pacmans location
        Returns:
            mdp: MDPSolver object.
        '''
    def getMDP(self, currLoc):

        states = [n for n in self.world.nodes if n.status != 'w']
        currState = next((n for n in states if n.getLocation() == currLoc), None)

        mdp = MDPSolver(currState, states)
        mdp.valueIteration()

        return mdp

    '''
    Sets up the ghosts in the map. calls the ghost affect for each ghost

    Args:
        @nodes: list of node objects

    '''
    def buildGhosts(self, nodes):

        gEffectDict = {}

        self.ghostRadius = 2 if self.world.boundaries[0][1] > 10 else 0

        for ghost in self.world.ghostStates:
            if ghost[1] == 0:
                gEffectDict = self.setGhostEffect(ghost[0], gEffectDict)

        for ghost in self.world.ghosts:
            if ghost in gEffectDict:
                gEffectDict.pop(ghost)

        for k, v in gEffectDict.items():

            n = self.world.getNode(k, nodes)
            n.reward = v
            n.status = 'gz'


    ''''
    Builds the ghost safety zone for pacman and sets corresponding reward
    values for states based on number of steps from the ghost.
    Args:
        @ghost: tuple containg location of ghost
        @locDict: dictonary contain reward values for the locations in the safety zone
    Returns:
        @locDict: dictonary contain reward values for the locations in the safety zone
    '''
    def setGhostEffect(self, ghost, locDict):

        distancePenalty = {1: -300, 2:-200, 3:-100, 4:-50}

        safeArea = self.getGhostArea(ghost)
        safeArea.remove(ghost)

        for loc in safeArea:

            dist = abs(loc[0] - ghost[0]) + abs(loc[1]-ghost[1])
            if dist not in distancePenalty:
                continue

            if loc not in locDict:
                locDict[loc] =  distancePenalty[dist]
            else:
                locDict[loc] +=  distancePenalty[dist]

        return locDict

    '''
    builds ghost area based on self.ghostRadius and returns List
    if tuples with locations in the area.
    Args:
        @ghostLoc: tuple containing location of the ghost.
    Returns:
        ghostArea: list of tuples with locations in the area
    '''
    def getGhostArea(self, ghostLoc):

        mapping = [(x,y) for x in range(self.ghostRadius + 1) for y in range(self.ghostRadius + 1)]
        quad = lambda z: ((z[0]*1, z[1] * 1), (z[0]*(-1), z[1] * 1), (z[0]*(1), z[1] * (-1)), (z[0]*-1, z[1] * -1))
        mapping = list(set([n for loc in map(quad, mapping) for n in loc]))

        ghostLocLst = [ghostLoc for i in range(len(mapping))]
        ghostArea = list(map(lambda x, y: (max(x[0]+y[0],0), max(x[1]+y[1],0)), mapping, ghostLocLst))

        ghostArea = list(filter(lambda x: x not in self.world.walls, ghostArea))

        ghostArea = [g for g in ghostArea if g[0] < self.world.boundaries[0][1] and g[1] < self.world.boundaries[1][1]]
        ghostArea = [g for g in ghostArea if g[0] > self.world.boundaries[0][0] and g[1] > self.world.boundaries[1][0]]

        return ghostArea


'''
This class contains logic for mapping the Pacworld based on the game state information.
Uses the Node class to build all the nodes in the map and sets the status for each node
based on where the ghosts, pacman and food is.
'''
class PacWorld():

    def __init__(self, state):
        self.ghostRadius = None
        self.corners = api.corners(state)
        self.walls = api.walls(state)
        self.food = api.food(state)
        self.ghosts = api.ghosts(state)
        self.pacMan = api.whereAmI(state)
        self.ghostStates = api.ghostStates(state)
        self.boundaries = None
        self.nodes = self.setNodes()
        self.buildPacWorld()

    '''
    Takes a location and list of node objects and returns corresponding
    node object
    Args:
        @location: tuple of location coordinates
    Returns:
        node object
    '''
    def getNode(self, location, nodes):
        return next((n for n in nodes if n.getLocation() == location), None)

        '''
        Calls methods to Build the Pacman world
        '''
    def buildPacWorld(self):
        self.setStatus(self.nodes)

        '''
        Returns a list of node object that make up the nodes of the map.
        Returns:
            nodes: List of node objects.
        '''
    def setNodes(self):

        edges = [loc for loc in self.corners if loc[0] and loc[1] != 0][0]

        if edges[0] < 10:
            self.boundaries = [(0, edges[0]), (0, edges[1])]
        else:
            self.boundaries = self.reduceMapMode(edges)

        nodes = [Node(x,y) for y in range(self.boundaries[1][0], self.boundaries[1][1] + 1)
                                                for x in range(self.boundaries[0][0], self.boundaries[0][1] + 1)]
        return nodes

        '''
        sets the status of each node to one of the following
        'o': for open node
        'f': for food node
        'g': for ghost node
        'w': for wall node
        '''
    def setStatus(self, nodes):

        occupiedNodes = self.food + self.ghosts + self.walls

        openNodes = [n.getLocation() for n in self.nodes if n.getLocation() not in occupiedNodes]

        status = {'o': openNodes, 'f': self.food, 'g': self.ghosts, 'w': self.walls}
        for n in nodes:
            for k, v in status.items():
                if n.getLocation() in v:
                    n.status = k


    '''
    reduces the size of the map, calculates new boundaries based on
    where nodes of interest (ghosts, food and pacman) are.
    Args:
        @edges: returns boundaries
    Returns:
        boundaries: list of tuple containing the new edges for x and y axis.
    '''
    def reduceMapMode(self, edges):

        ghosts = [(int(g[0]), int(g[1])) for g in self.ghosts]
        objects = ghosts + self.food + [self.pacMan]
        maxY = max([o[1] for o in objects])
        minY = min([o[1] for o in objects])
        maxX = max([o[0] for o in objects])
        minX = min([o[0] for o in objects])

        maxX += 1 if maxX != edges[0] else edges[0]
        maxY += 1 if maxY != edges[1] else edges[1]
        minX -= 1 if minX != 0 else 0
        minY -= 1 if minY != 0 else 0

        boundaries = [(minX, maxX), (minY, maxY)]

        return boundaries

'''
Defines methods and attribute for a node object. nodes
have a utility, status, reward, x and y coordinated
associated with them.
'''
class Node():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.status = None
        self.utility = None
        self.reward = 0
        self.displayValue = ''

        '''setter method. Sets status for the node
        Args:
            @status: character representing status
        '''
    def setStatus(self, status):
        self.status = status

        '''setter method. Sets utility for the node
        Args:
            @utility: integer value
        '''
    def setUtility(self, utility):
        self.utility = utility

        '''
        getter method. returns tuple of x, y coordinates.
        '''
    def getLocation(self):
        return (self.x, self.y)
