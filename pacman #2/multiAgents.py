# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        #print ("bestScore: ", bestScore)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #print ("bestIndices: ", bestIndices)
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #print ("chosen index: ", chosenIndex)

        "Add more of your code here if you want to"
        #print("legal: ", legalMoves[chosenIndex])
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #print ("action: ", action)
        if len(newFood.asList()) == currentGameState.getFood().count():
            #print("new food: ", newFood.asList())
            score = 99999
            for f in newFood.asList():
                if manhattanDistance(f , newPos) < score :
                    score = manhattanDistance(f, newPos)
                    #print ("dis1: ", score)
        else:
            score = 0
        for ghost in newGhostStates:  # the impact of ghost surges as distance get close
            #dis -= 5 ** (2 - manhattanDistance(ghost.getPosition(), newPos))
            #print("md: ", 2 + manhattanDistance(ghost.getPosition(), newPos))
            score += 5 ** (2 - manhattanDistance(ghost.getPosition(), newPos))
            #print ("ghost dis: ", dis)
        #print ("dis: ", score)
        return -score

        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimax(self, state, depth, agent = 0, maxing = True):

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        actions = state.getLegalActions(agent)

        if maxing == True:
            scores = [self.minimax(state.generateSuccessor(agent, action), depth - 1, 1, False)[0] for action in actions]
            bestScore = max(scores)
            bestIndices = [i for i in range(len(scores)) if scores[i] == bestScore]
            return bestScore, actions[bestIndices[0]]

        elif maxing == False:
            scores = []

            if agent == state.getNumAgents() - 1:
                scores = [self.minimax(state.generateSuccessor(agent, action), depth - 1, 0, True)[0] for action in actions]

            else:
                scores = [self.minimax(state.generateSuccessor(agent, action), depth, agent + 1, False)[0] for action in actions]

            bestScore = min(scores)
            bestIndices = [i for i in range(len(scores)) if scores[i] == bestScore]
            return bestScore, actions[bestIndices[0]]


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"   
        return self.minimax(gameState, self.depth * 2, 0, True)[1]

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        depth = 0
        alpha = float('-inf')
        beta = float('inf')
        return self.getMaxValue(gameState, alpha, beta, depth)[1]

    def getMaxValue(self, gameState, alpha, beta, depth, agent = 0):
        actions = gameState.getLegalActions(agent)

        if not actions or gameState.isWin() or depth >= self.depth:
            return self.evaluationFunction(gameState), Directions.STOP

        successorCost = float('-inf')
        successorAction = Directions.STOP

        for action in actions:
            successor = gameState.generateSuccessor(agent, action)

            cost = self.getMinValue(successor, alpha, beta, depth, agent + 1)[0]

            if cost > successorCost:
                successorCost = cost
                successorAction = action

            if successorCost > beta:
                return successorCost, successorAction

            alpha = max(alpha, successorCost)

        return successorCost, successorAction

    def getMinValue(self, gameState, alpha, beta, depth, agent):
        actions = gameState.getLegalActions(agent)

        if not actions or gameState.isLose() or depth >= self.depth:
            return self.evaluationFunction(gameState), Directions.STOP

        successorCost = float('inf')
        successorAction = Directions.STOP

        for action in actions:
            successor = gameState.generateSuccessor(agent, action)

            cost = 0

            if agent == gameState.getNumAgents() - 1:
                cost = self.getMaxValue(successor, alpha, beta, depth + 1)[0]
            else:
                cost = self.getMinValue(successor, alpha, beta, depth, agent + 1)[0]

            if cost < successorCost:
                successorCost = cost
                successorAction = action

            if successorCost < alpha:
                return successorCost, successorAction

            beta = min(beta, successorCost)

        return successorCost, successorAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.getActionExpectimax(gameState, self.depth, 0)[1]

    
    def getActionExpectimax(self, gameState, depth, agentIndex):

        agentNum = gameState.getNumAgents()
        if depth == 0 or gameState.isWin() or gameState.isLose():
            eResult = self.evaluationFunction(gameState)
            return (eResult, '')

        else:
            maxAct = ''
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

            if agentIndex == agentNum - 1:
                depth -= 1

            if agentIndex == 0:
                maxAlp = float('-inf')

            else:
                maxAlp = 0

            maxAct = ''
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

            for action in gameState.getLegalActions(agentIndex):
                gState = gameState.generateSuccessor(agentIndex, action)
                result = self.getActionExpectimax(gState, depth, nextAgentIndex)

                if agentIndex == 0:
                    if result[0] > maxAlp:
                        maxAlp = result[0]
                        maxAct = action
                else:
                    maxAlp += 1.0/len(gameState.getLegalActions(agentIndex)) * result[0]
                    maxAct = action

        return (maxAlp, maxAct)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Score increases as PacMan eats food and capsules. If a ghost is encountered,
    within a distance of 2 from PacMan, the score will become extremely low, making sure that 
    pacman stays away from the ghosts.

    """
    "*** YOUR CODE HERE ***"
    
    num = 0
    minDistance = float('inf')
    minDistanceBool = False
    score = currentGameState.getScore()

    #print("restart")
    for foodPosition in currentGameState.getFood().asList():
        foodDistance = util.manhattanDistance(currentGameState.getPacmanPosition(), foodPosition)
        if foodDistance < minDistance:
            minDistance = foodDistance
            #print("minDistance: ", minDistance)
            minDistanceBool = True

    if minDistanceBool == True:
        num += minDistance

    num += 900*currentGameState.getNumFood()
    #print('eNum, get num food: ', eNum)
    num += 9*len(currentGameState.getCapsules())
    #print('eNum, getCapsules: ', eNum)

    for ghostPos in currentGameState.getGhostPositions():
        ghostDistance = util.manhattanDistance(currentGameState.getPacmanPosition(), ghostPos)
        if ghostDistance < 2:
            num = float('inf')

    num -= 9*score
    #print("current score: ", score)
    #print("return eNum: ", -eNum)
    return -num

better = betterEvaluationFunction

