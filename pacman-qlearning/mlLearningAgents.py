# mlLearningAgents.py
# Gregory Verghese/26-mar-2018
#code template provided by Simon Parsons
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
import numpy
import random


'''
This class implements our agent that tries to win Pacman by learning using a reinforcement technique.
This particular implementation calls a SARSA class to train over self.numTraining games before it attempts
to play as a test.
'''
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.15, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.score = 0
        self.QLearnObj = None


    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts


    '''
    getAction method controls flow of pacmans decision making for the next move.
    Gets required information from the state object, ocntrols training vs playing.
    Either calls goLearn or gets action from QLearnObj to find next move.
    Args:
        @state: gamestate object containing pacman current state info
    Returns:
       self.getMove(action): string legal action ('North', 'South', 'East', 'West']
    '''
    def getAction(self, state):

        legal = state.getLegalPacmanActions()

        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        actions = [self.getVector(move) for move in legal]

        position = state.getPacmanPosition()
        food = state.getFood()
        ghosts = state.getGhostPositions()
        totalScore = state.getScore()
        reward = totalScore - self.score

        self.score = totalScore

        if self.getEpisodesSoFar() < self.numTraining:
            action = self.goLearn(reward, actions, position, ghosts, food)
            return self.getMove(action)

        state = self.QLearnObj.buildState(position, ghosts, food)
        maxQ, action =  self.QLearnObj.getQandAction(state)

        return self.getMove(action)

    
    '''
    goLearn method sets up intial state and qTable if it is the first
    game of training and returns a random move from set of legal moves.
    Calls the SARSA class to learn if not the first game.
    Args:
        @reward: numerical reward used in Q-update. Difference between total score (Score_current - Score_prev)
        @actions: list of vectorized x, y tuples of legal actions
        @position: x, y tuple of pacmans position
        @ghosts: list of tuples of ghost positions
        @food: list of tuples of food positions
    Returns:
        action: x, y tuple of next action
    '''
    def goLearn(self, reward, actions, position, ghosts, food):

        if self.episodesSoFar == 0:
            table = QTable()
            self.QLearnObj = SARSA(self.alpha, self.epsilon, self.gamma, table)

            state = self.QLearnObj.buildState(position, ghosts, food)
            table.qTable[state] = {a: 0 for a in actions}

            action = random.choice(actions)

            self.QLearnObj.prevState = state
            self.QLearnObj.prevAction = action
            self.QLearnObj.reward = reward
            
            return action

        action = self.QLearnObj.learn(reward, actions, position, ghosts, food)

        return action


    '''
    translates string move into vectorized x, y tuple action. Taken from Gregory Verghese
    artificial intelligence coursework.
    Args:
        @move: string move
    Returns:
        vectorDict[move]: a x, y tuple vectorized action
    '''
    def getVector(self, move):
        vectorDict = {'North': (0, 1), 'South': (0, -1), 'West': (-1, 0), 'East': (1, 0)}

        return  vectorDict[move]


    '''
    translates x, y action vector in string move. Taken from Gregory Verghese artifical intelligence
    coursework.
    Args:
        @vector: a x, y tuple vectorized action
    Returns:
        moveDict[move]: string move  
    '''
    def getMove(self, vector):
        moveDict = {(0, 1): 'North', (0, -1): 'South', (-1, 0): 'West', (1, 0): 'East'}

        return  moveDict[vector]


    '''
    final method called when games ends (win or die). Calls final Q-update
    Args:
        @state: pacman gamestate object
    '''
    def final(self, state):

        print "A game just ended!"

        # Keep track of the number of games played, and set learningedges = [loc for loc in self.corners if loc[0] and loc[1] != 0][0]
        # parameters to zero when we are done with the pre-set number
        # of training episodes

        position = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        food= state.getFood()
        totalScore = state.getScore()
        reward = totalScore - self.score
        actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        self.QLearnObj.learn(reward, actions, position, ghosts, food)

        self.incrementEpisodesSoFar()

        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


'''
SARSA class outlines implementation SARSA learning. Attributes, alpha, epsilon and gamma from
Q-update and a table object for Q-table values.
'''
class SARSA():

    def __init__(self, alpha, epsilon, gamma, table):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.QTable = table
        self.prevState = None
        self.prevAction = None
        self.reward = None


    '''
    learn method builds current state (different from state object)
    and gets the update action based on epsilonGreedy and sets Q-values
    if the state is new or calls update.
    Args:
        @reward: numerical reward used in Q-update. Difference between total score (Score_current - Score_prev)
        @actions: list of vectorized x, y tuples of legal actions
        @position: x, y tuple of pacmans position
        @ghosts: list of tuples of ghost positions
        @food: list of tuples of food positions
    Returns:
        newAction: x, y tuple representing vetorized action.
    '''
    def learn(self, reward, actions, position, ghosts, food):

        state = self.buildState(position, ghosts, food)

        if state not in self.QTable.qTable:
            self.QTable.qTable[state] = {a: 0 for a in actions}

        newAction = self.epsilonGreedy(state, actions)

        self.update(newAction, state)
        self.reward = reward

        self.prevState = state
        self.prevAction = newAction

        return newAction


    '''
    epsilonGreedy controls exploration vs exploitation. Compares
    randomly generated number against epsilon and chooses action
    accordingly (randomly vs max action based on Q-values)
    Args:
        @state: current sarsa state being explored
        @actions: list of x, y tuples of actions
    Returns:
        bestAction: x, y tuple action
        
    '''
    def epsilonGreedy(self, state, actions):

        randNum = random.random()

        if randNum < self.epsilon:
            bestAction = random.choice(actions)
        elif randNum >= self.epsilon:
            currQ, bestAction = self.getQandAction(state)

        return bestAction


    '''
    SARSA Q-update
    Args:
        @newAction: x, y tuple of update action
        @state: current sarsa state being explored
    '''
    def update(self, newAction, state):

        q = self.QTable.qTable[self.prevState][self.prevAction]
        qNew = self.QTable.getQ(state, newAction)

        q = q + self.alpha*(self.reward + self.gamma*qNew - q)
        self.QTable.setQ(self.prevState, self.prevAction, q)


    '''
    build sarsa state based on position ghosts and food
    Args:
        @position: x, y tuple of pacmans current position
        @ghosts: lsit of x, y tuple of ghosts position
        @food: list of x, y tuple of food positions
    Returns:
        state: SARSA state to add to table
    '''
    def buildState(self, position, ghosts, food):

        food = list(food)
        ghosts2 = [g for g in ghosts]
        food = [(x, y) for x in range(len(food)) for y in range(len(food[0])) if food[x][y]==True]

        state1 = ghosts2 + food
        state1.insert(0, position)

        state = tuple(state1)

        return state


    '''
    gets the state-action pair with max Q-value from table
    Args:
        @state: current sarsa state being explored
    Returns:
        maxQ: maximum Q value
        action: action corresponding to max q value for given state
    '''
    def getQandAction(self, state):

        qs = self.QTable.qTable[state].items()
        stateActionsQs = zip(*qs)
        stateActions = stateActionsQs[0]
        stateQs = stateActionsQs[1]
        maxQ = max(stateQs)
        action = stateActions[stateQs.index(maxQ)]

        return maxQ, action


'''
outlines Q-table implementation. self.qTable stores state-action pair Q values
'''
class QTable():

    def __init__(self):

        self.actions = {}
        self.qTable = {}


    '''
    setter method
    Args:
        @state: current sarsa state being explored
        @action: action
    '''
    def setQ(self, state, action, q):
        self.qTable[state][action] = q

    '''
    getter method
    Args:
        @state: current sarsa state being explored
        @action: action
    returns:
        self.qTable[state][action]: q value for state action pair       
    '''
    def getQ(self, state, action):
        return self.qTable[state][action]
