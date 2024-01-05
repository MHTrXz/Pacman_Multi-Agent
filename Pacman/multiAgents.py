import random
import time
import util
from util import manhattanDistance
from game import Agent
from pacman import GameState


def scoreEvaluationFunction(currentGameState: GameState):
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="4", time_limit="6"):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def heuristicFood(self, gameState: GameState):
        position = gameState.getPacmanPosition()
        distance = float('inf')
        for i in gameState.getFood().asList():
            distance = min(distance, manhattanDistance(position, i))
        return 50 - distance

    def heuristicLastFood(self, gameState: GameState):
        foods = gameState.getFood().asList()
        if len(foods) == 1:
            return 100
        else:
            return 1

    def heuristicGhost(self, gameState: GameState):
        position = gameState.getPacmanPosition()
        distance = []
        for i in gameState.getGhostStates():
            if i.scaredTimer > 4:
                distance.append(-manhattanDistance(position, i.getPosition()))
            else:
                distance.append(manhattanDistance(position, i.getPosition()))
        return 20 + min(distance)

    def heuristicScaredGhost(self, gameState: GameState):
        position = gameState.getPacmanPosition()
        distance = float('inf')
        status = False
        for i in gameState.getGhostStates():
            if i.scaredTimer > 4:
                status = True
                distance = min(distance, manhattanDistance(position, i.getPosition()))
        return 20 - distance if status else 0

    def Handler(self, gameState, agentIndex, depth, startTime, alpha, beta, currentGameState):
        oldFoods = gameState.getNumFood()
        if gameState.isWin():
            return 100000, None
        elif gameState.isLose():
            return -100000, None
        elif depth == self.depth or (time.time() - startTime) >= self.time_limit:
            score = gameState.getScore()
            return score + (score / (self.depth)) * (oldFoods - gameState.getNumFood()) + (
                    self.heuristicLastFood(currentGameState) * self.heuristicFood(gameState)) + (
                               0.3 * self.heuristicGhost(gameState)), None
        else:
            if agentIndex == 0:
                lState = self.calcValue(gameState, True, depth, agentIndex, startTime, alpha, beta, currentGameState)
                return lState[0] - 1, lState[1]
            else:
                lState = self.calcValue(gameState, False, depth, agentIndex, startTime, alpha, beta, currentGameState)
                return lState[0], lState[1]

    def calcValue(self, gameState: GameState, isMax: bool, depth, agentIndex, startTime, alpha, beta, currentGameState):
        score = float('-inf') if isMax else float('inf')
        actions = []
        legalAction = gameState.getLegalActions(agentIndex)
        for action in legalAction:
            successor = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextAgentDepth = depth + 1 if (agentIndex + 1) == gameState.getNumAgents() else depth
            nextAgentsScore = \
            self.Handler(successor, nextAgent, nextAgentDepth, startTime, alpha, beta, currentGameState)[0]
            if isMax:
                if score < nextAgentsScore:
                    score = nextAgentsScore
                    actions = [action]
                elif score == nextAgentsScore:
                    actions = actions + [action]
            else:
                if score > nextAgentsScore:
                    score = nextAgentsScore
                    actions = [action]
                elif score == nextAgentsScore:
                    actions = actions + [action]
            if isMax:  # AlphaBetaPruning
                if score > beta:
                    return score, random.choice(actions)
            else:
                if score < alpha:
                    return score, random.choice(actions)
        return score, random.choice(actions)

    def getAction(self, gameState: GameState):
        score, action = self.Handler(gameState, 0, 0, time.time(), float("-inf"), float("inf"), gameState)

        nextState = gameState.generateSuccessor(0, action)

        if nextState.isWin() or nextState.isLose():  # output
            self.output(nextState)

        return action

    def output(self, nextState: GameState):
        output = "Pacman Game\n"
        if nextState.isWin():
            output += "Result: Win\n"
        elif nextState.isLose():
            output += "Result: Lose\n"
            output += "Foods: " + str(nextState.getNumFood())
        output += "Score: " + str(nextState.getScore()) + "\n"
        output += "Agents: " + str(nextState.getNumAgents()) + "\n"
        output += "Depth: " + str(self.depth) + "\n"
        output += "Ghosts: " + str(nextState.getNumAgents() - 1) + "\n"
        output += "----------------\n"
        f = open("../Outputs/The_Phoenix-UIAI4021-PR4.txt", 'a')
        f.write(output)
        f.close()
