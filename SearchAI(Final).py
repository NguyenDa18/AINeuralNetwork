__author__ = 'Danh Nguyen and Jacob Kirby'

import random, time
from Player import *
from Constants import *
from Construction import *
from Ant import *
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *
import math

#Python representation of infinity
INFINITY = float("inf")

# dictionary representation of a node in the search tree
treeNode = {
    # the Move that would be taken in the given state from the parent node
    "move"    : None,
    # the state that would be reached by taking the above move
    "state"   : None,
    # an evaluation of the next_state
    "score"   : 0.0,
    # a reference to the parent node
    "parent"  : None
}

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        # a depth limit for the search algorithm
        self.depthLimit = 3
        super(AIPlayer,self).__init__(inputPlayerId, "TEST BRUISER CRUISER")

        ############# NEURAL NETWORK CODE #############
        self.weightList = [0.9836148983403992, 0.27973053332555825, 0.8760461326118066, -0.33343, 1.9102820505439173,
                           0.9828365968678142, 0.11864432002059076, 0.17976855640452372, 0.1, 0.5836148983403951,
                           0.37973053332555756, 0.9760461326118066, 0.7, 0.8769099929280543, 0.4172951874936439,
                           -0.3959125983634327, -0.1, 0.5338010375505289, 0.5498737299684558, 0.8124731441831693, 0.3,
                           0.1, 0.2, 0.9, -0.7, 2.810282050543916, 0.4268403577148738, 2.0795444233961744, 0.3]
        #biases for each node
        self.biases = []

        #learning rate
        self.ALPHA = .3

        #outputs for each node
        self.outputs = [0.0, 0.0, 0.0]



    ##
    #getPlacement
    #Description: The getPlacement method corresponds to the
    #
    #Parameters:
    #   currentState - The current state of the game at the time the Game is
    #       requesting a placement from the player.(GameState)
    #
    #Return: If setup phase 1: list of ten 2-tuples of ints -> [(x1,y1), (x2,y2),?,(x10,y10)]
    #       If setup phase 2: list of two 2-tuples of ints -> [(x1,y1), (x2,y2)]
    #
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]


    ##
    #getMove
    #Description: The getMove method corresponds to the play phase of the game
    #and requests from the player a Move object. All types are symbolic
    #
    #Parameters:
    #   currentState - The current state of the game at the time the Game is
    #       requesting a move from the player.(GameState)
    #
    #Return: Move(moveType [int], coordList [list of 2-tuples of ints], buildType [int]
    #
    def getMove(self, currentState):
        # save our id and inventory
        me = currentState.whoseTurn
        myInv = getCurrPlayerInventory(currentState)

        #create the initial node to analyze
        initNode = self.makeNode(None, currentState, None)

        ###########################################################
        ############### HANDLE ALPHA BETA PRUNING #################
        ###########################################################
        bestNode = self.find_max(initNode, -INFINITY, INFINITY, 0)
        while bestNode["parent"]["parent"] is not None:
            bestNode = bestNode["parent"]

        return bestNode["move"]

    ##
    #getAttack
    #Description: The getAttack method is called on the player whenever an ant completes
    #
    #Return: A coordinate that matches one of the entries of enemyLocations. ((int,int))
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##################################################################################
    ########################### HW3: MINIMAX SEARCH #################################
    ##################################################################################
    ##
    # makeNode
    # Description: Creates a node with values set based on parameters
    #
    # Parameters:
    #   self - The object pointer
    #   move - The move that leads to the resultingState
    #   resultingState - The state that results from making the move
    #   parent - The parent node of the node being created
    #
    # Returns: A new node with the values initialized using the parameters
    #
    def makeNode(self, move, resultingState, parent):
        # Create a new node using treeNode as a model
        nextNode = treeNode.copy()
        # set the move
        nextNode["move"] = move
        # set the state that results from making the move
        nextNode["state"] = resultingState
        # set the value of the resulting state
        nextNode["score"] = self.neuralNetEval(resultingState)
        # store a reference to the parent of this node
        nextNode["parent"] = parent

        return nextNode

    ##
    # find_max
    # Description: returns the best move our player can make from the current state
    #
    # Parameters:
    #   self - the object pointer
    #   node - the current node, before any moves are explored
    #   alpha - the alpha value, the value of our best move
    #   beta - the value of the opponent's best move
    #   currentDepth - the current depth of the node from the initial node
    #
    # Returns: the move which benefits the opposing player the least (alpha).
    #
    def find_max(self, node, alpha, beta, currentDepth):
        value = -INFINITY

        # base case, depthLimit reached, return the value of the currentState
        if currentDepth == self.depthLimit:
            return node
        state = node["state"]

        # holds a list of nodes reachable from the currentState
        nodeList = []
        # loop through all legal moves for the currentState
        for move in listAllLegalMoves(state):
            # don't evaluate for the queen unless we need to build a worker
            if move.moveType == MOVE_ANT:
                initialCoords = move.coordList[0]
                if ((getAntAt(state, initialCoords).type == QUEEN) and
                    len(state.inventories[state.whoseTurn].ants) >= 2):
                        continue

            # get the simulated state if the move were made
            resultingState = self.getNextStateAdversarial(state, move)

            #create a nextNode for the simulated state
            nextNode = self.makeNode(move, resultingState, node)

            # if a goal state has been found, stop evaluating other branches
            if nextNode["score"] == 1.0:
                return nextNode
            nodeList.append(nextNode)

        #sort nodes from greatest to least
        sortedNodeList = sorted(nodeList, key=lambda k: k['score'], reverse=True)

        #holds a reference to the current best node to move to
        optimalValNode = None

        #if it is our turn
        if (self.playerId == state.whoseTurn):
            for selectNode in sortedNodeList:
                maxValNode = self.find_max(selectNode, alpha, beta, currentDepth + 1)
                #if we're in find_max, stay in find_max
                if value < maxValNode["score"]:
                        optimalValNode = maxValNode
                        value = maxValNode["score"]
                if value >= beta:
                    return maxValNode
                alpha = max(alpha, value)
        #If it is the opponent's turn
        else:
            sortedNodeList = sorted(nodeList, key=lambda k: k['score'])
            for selectNode in sortedNodeList:
                maxValNode = self.find_min(selectNode, alpha, beta, currentDepth + 1)
                #if it's the opponent's turn and they're in find_max, change to find_min
                if value < maxValNode["score"]:
                       optimalValNode = maxValNode
                       value = maxValNode["score"]
                if value >= beta:
                    return maxValNode
                alpha = max(alpha, value)

        return optimalValNode

    ##
    # find_min
    # Description: returns the best move our opponent can make from the current state
    #
    # Parameters:
    #   self - the object pointer
    #   node - the current node, before any moves are explored
    #   alpha - the alpha value, value of our best move
    #   beta - the value of the opponent's best move
    #   currentDepth - the current depth of the node from the initial node
    #
    # Returns: the move which benefits the opposing player the least (alpha).
    #
    def find_min(self, node, alpha, beta, currentDepth):
        # base case, depthLimit reached, return the value of the currentState
        if currentDepth == self.depthLimit:
            return node
        state = node["state"]
        value = INFINITY

        # holds a list of nodes reachable from the currentState
        nodeList = []
        # loop through all legal moves for the currentState
        for move in listAllLegalMoves(state):
            # don't evaluate for the queen unless we need to build a worker
            if move.moveType == MOVE_ANT:
                initialCoords = move.coordList[0]
                if ((getAntAt(state, initialCoords).type == QUEEN) and
                    len(state.inventories[state.whoseTurn].ants) >= 2):
                        continue

            # get the simulated state if the move were made
            resultingState = self.getNextStateAdversarial(state, move)
            #create a nextNode for the simulated state
            nextNode = self.makeNode(move, resultingState, node)

            # if a goal state has been found, stop evaluating other branches
            if nextNode["score"] == 0.0:
                return nextNode
            nodeList.append(nextNode)

        #sort nodes from least to greatest
        sortedNodeList = sorted(nodeList, key=lambda k: k['score'])

        #holds a reference to the current best node to move to
        optimalValNode = None

        #if it is our turn
        if (self.playerId == state.whoseTurn):
            for selectNode in sortedNodeList:
                minValNode = self.find_max(selectNode, alpha, beta, currentDepth + 1)
                #if we're in find_max, stay in find_max
                if value > minValNode["score"]:
                        optimalValNode = minValNode
                        value = minValNode["score"]
                if value <= alpha:
                    return minValNode
                beta = min(beta, value)
        #If it is the opponent's turn
        else:
            for selectNode in sortedNodeList:
                minValNode = self.find_min(selectNode, alpha, beta, currentDepth + 1)
                #if the opponent is in find_max, change to find_min
                if value > minValNode["score"]:
                       optimalValNode = minValNode
                       value = minValNode["score"]
                if value <= alpha:
                    return minValNode
                beta = min(beta, value)

        return optimalValNode

    ##
    # getNextStateAdversarial
    # Description: The getNextStateAdversarial method looks at the current state and simulates a move,
    # taking into consideration that the agents have to take turns
    # Parameters:
    #   currentState -the state our game is in
    #   move -move we will simulate
    #
    # Return: The resulting state after move is made
    #
    def getNextStateAdversarial(self, currentState, move):
        nextState = currentState.fastclone()
        # Find our inventory and enemies inventory
        clonedInventory = None
        enemyInventory = None
        if nextState.inventories[PLAYER_ONE].player == self.playerId:
            clonedInventory = nextState.inventories[PLAYER_ONE]
            enemyInventory = nextState.inventories[PLAYER_TWO]
        else:
            clonedInventory = nextState.inventories[PLAYER_TWO]
            enemyInventory = nextState.inventories[PLAYER_ONE]

        # Check if move is a build move, ignore it
        if move.moveType == BUILD:
            pass
        elif move.moveType == MOVE_ANT:
            startCoord = move.coordList[0]
            finalCoord = move.coordList[-1]
            # Update the coordinates of the ant to move
            for ant in clonedInventory.ants:
                if ant.coords == startCoord:
                    # update the ant's coords
                    ant.coords = finalCoord
                    adjacentTiles = listAdjacent(ant.coords)
                    for adj in adjacentTiles:
                        if getAntAt(nextState, adj) is not None:
                            closeAnt = getAntAt(nextState, adj)
                            if closeAnt.player != nextState.whoseTurn:
                                closeAnt.health = closeAnt.health - UNIT_STATS[ant.type][ATTACK]
                                if closeAnt.health <= 0:
                                    enemyAnts = enemyInventory.ants
                                    for enemy in enemyAnts:
                                        if closeAnt.coords == enemy.coords:
                                            enemyInventory.ants.remove(enemy)
                                break
                            break
        #copied from getNextStateAdversarial in AIPlayerUtils
        elif move.moveType == END:
            for ant in clonedInventory.ants:
                ant.hasMoved = False
            nextState.whoseTurn = 1 - currentState.whoseTurn;

        return nextState


    # ----------------------------------------------------------------------------------------------------------------#
    # HW5: NEURAL NETS#
    # ----------------------------------------------------------------------------------------------------------------#
    ##
    # initNeuralNet
    #
    # Description: Takes GameState and evaluates it with a neural net
    # Generates weight list with random numbers first
    #
    def initNeuralNet(self):
        # for each node connection, assign a weight
        for i in range(0, 29):
            weight = random.uniform(-3, 3)
            self.weightList.append(weight)

            # #
            # gfx
            # Description: applies the 'g' function used by our neural network
            #
            # Parameters:
            #   x - the variable to apply g to
            #
            # Return: g(x)
            # #

    def gfx(self, x):
        return 1 / (1 + math.exp(-x))


        ##
        # neuralNetEval
        #
        # Description: Takes GameState and evaluates it with a neural net
        # Return: number b/w 0 and 1 that depicts how good the state is
        #

    def neuralNetEval(self, currentState):

        # Grab the playerIDs
        oppId = 0 if self.playerId == 1 else 1
        #
        foodList = []
        self.myFood = None
        # self.myTunnel = None
        # self.myHill = None

        # ##### OUR STUFF #####
        inventory = getCurrPlayerInventory(currentState)

        # Winning/losing conditions
        if self.hasWon(currentState, oppId):  # we lose
            value = 0
            return value
        elif self.hasWon(currentState, self.playerId):  # we win
            value = 1
            return value


        if self.myFood is None:
            foods = getConstrList(currentState, None, (FOOD,))
            self.myFood = foods[0]
            # find the food closest to the tunnel
            bestDistSoFar = 1000  # i.e., infinity
            for food in foods:
                if food.coords[1] < 4:
                    foodList.append(food)

        # Number of nodes (constant)
        numNetNodes = 5

        # Array to store inputs
        net_inputs = []

        net_inputs.append(self.evalQueen(inventory, currentState, foodList))
        net_inputs.append(self.evalWorkers(inventory, currentState))
        net_inputs.append(self.evalWorkerCarrying(inventory, currentState))
        net_inputs.append(self.evalWorkerNotCarrying(inventory, currentState))
        net_inputs.append(self.evalFoodCount(inventory, currentState))
        print "net_inputs ", net_inputs
        # TESTING
        scoreSum = 0.0
        for input in net_inputs:
            scoreSum += input

        target = 1.0

        net_outputs = self.propagateNeuralNetwork(net_inputs)
        print net_outputs
        self.backPropagateNeuralNetwork(target, net_outputs, net_inputs)
        error = net_outputs[numNetNodes - 1] - target

        print "Error: \n"
        print error

        # We need our workers to be moving towards some food.
        numWorkers = 0

        return scoreSum

    def propagateNeuralNetwork(self, inputs):
        numNetNodes = 5
        nodeValues = [0] * numNetNodes
        output = numNetNodes - 1
        count = 0
        # nodeVals = [0] * numNetNodes

        # bias generation = 4 biases for the inputs and 1 for output
        for count in range(0, numNetNodes):
            nodeValues[count] = self.weightList[count]
            count += 1

        # 25 weights from inputs to hidden nodes
        for i in range(0, len(inputs)):
            for j in range(0, numNetNodes - 1):
                nodeValues[j] += inputs[i] * self.weightList[count]
                count += 1

        # mults. hidden nodes by g fxn
        for node in range(numNetNodes - 1):
            nodeValues[node] = self.gfx(nodeValues[node])

        # 4 weights from hidden nodes to output
        for hiddenNode in range(0, numNetNodes - 1):
            nodeValues[output] += nodeValues[hiddenNode] * self.weightList[count]
            count += 1

        # calcs. g fxn of output
        nodeValues[output] = self.gfx(nodeValues[output])

        print "Node Values: "
        print nodeValues

        return nodeValues

        ##
        # backPropagateNeuralNetwork

    def backPropagateNeuralNetwork(self, target, outputs, inputs):
        numNetNodes = 5
        error = target - outputs[numNetNodes - 1]
        counter = (numNetNodes - 1) + (numNetNodes - 1) * len(inputs)

        delta = outputs[numNetNodes - 1] * (1 - outputs[numNetNodes - 1]) * error

        hiddenErrors = [0] * (numNetNodes - 1)
        hiddenDeltas = [0] * (numNetNodes - 1)

        for i in range(numNetNodes - 2):
            hiddenErrors[i] = self.weightList[counter + 1 + i] * delta
            hiddenDeltas[i] = outputs[i] * (1 - outputs[i]) * hiddenErrors[i]

        hiddenDeltas.append(delta)

        for weight in range(len(self.weightList) - 1):
            if weight < numNetNodes:
                nodeIndex = weight % numNetNodes
                input = 1
            elif weight > len(self.weightList) - numNetNodes:
                nodeIndex = numNetNodes - 1
                inputIdx = weight - (len(self.weightList) - numNetNodes)
                input = inputs[inputIdx]
            else:
                nodeIndex = (weight - 1) % (numNetNodes - 1)
                inputIdx = (weight - numNetNodes) / (numNetNodes - 1)
                input = inputs[inputIdx]

            # backprop equation
            self.weightList[weight] += self.ALPHA * hiddenDeltas[nodeIndex] * input

    # ##
    # # evalNodeList
    # #
    # # Description: Evaluates a node list and rates all of them
    # #
    # # Parameters:
    # #   nodes - The list of node objects to evaluate
    # #
    # # Returns: The average distance from 0.5 for each node
    # ##

    def evalNodeList(self, nodes):
        if len(nodes) == 0:
            return 0

        totalVal = 0
        for node in nodes:
            # print "Node Value: \n"
            # print node.value


            totalVal += (node.value - 0.5) / len(nodes)

        return totalVal

    def rateMoveOnDist(self, dist1, dist2):
        amount = .1 * ((float(dist1) - dist2) / dist1)
        return amount

        ##
        # expandState
        #
        # Definition: Expands the given state until the max depth has been reached
        #
        # Parameters:
        #   state - The current state of the game
        #   currentDepth - The current depth that we are at
        ##

    def expandCurrentState(self, state, currentDepth):
        # Find all moves that we can take and take them
        moveList = listAllLegalMoves(state)
        nodeList = []

        # Get all states for all moves
        for move in moveList:
            if move.moveType != END:
                antType = -1
                if getAntAt(state, move.coordList[0]) is not None:
                    antType = getAntAt(state, move.coordList[0]).type

                # We want to check if the queen is on our anthill
                onAntHill = False
                if move.coordList[len(move.coordList) - 1] == getConstrList(state, self.playerId, (ANTHILL,))[0].coords:
                    onAntHill = True

                # We want to check if our queen is on some food
                queenOnFood = False
                for food in getConstrList(state, None, (FOOD,)):
                    if move.coordList[len(move.coordList) - 1] == food.coords:
                        queenOnFood = True
                        break

                # We don't want to expand certain nodes, I'll let you read this to figure out which ones
                if (move.moveType == MOVE_ANT and antType != QUEEN and len(
                        move.coordList) > 1) or move.moveType == BUILD or \
                        (antType == QUEEN and not (onAntHill or queenOnFood) and move.moveType == MOVE_ANT and len(
                            move.coordList) > 1) or \
                        ((antType == QUEEN or antType == SOLDIER) and self.canAttack(state, move.coordList[0])):
                    nextState = getNextState(state, move)
                    val = self.neuralNetEval(state)
                    # print ("Shit \n")
                    # print val
                    nodeList.append(Node(move, nextState, self.neuralNetEval(state), state))

        # EDGE CASE: One of the ants can't move, so it bricks the system. Force an ant to move.
        if len(nodeList) == 0 and len(moveList) != 0:
            for desparateMove in moveList:
                nextState = getNextState(state, desparateMove)
                nodeList.append(Node(desparateMove, nextState, self.neuralNetEval(state), state))

        # Go deeper if we need to
        if currentDepth != self.depth - 1:
            for node in nodeList:
                # Only expand nodes that were reached by moving
                node.value += self.expandCurrentState(node.state, currentDepth + 1)

        # If we are in the middle of recursion, hand back the score. Otherwise, hand back the move to the
        # best option
        value = self.evalNodeList(nodeList)
        if currentDepth > 0:
            return value
        else:
            bestMove = None
            bestValue = 0
            for node in nodeList:
                if node.value > bestValue:
                    bestMove = node.move
                    bestValue = node.value

            return bestMove

    ##
    # hasWon
    #
    #   This function was copied from Game.py so we could use it here. Please
    #   refer to that class to see the description
    def hasWon(self, state, playerId):
        opponentId = (playerId + 1) % 2

        if ((state.phase == PLAY_PHASE) and
                ((state.inventories[opponentId].getQueen() == None) or
                     (state.inventories[opponentId].getAnthill().captureHealth <= 0) or
                     (state.inventories[playerId].foodCount >= FOOD_GOAL) or
                     (state.inventories[opponentId].foodCount == 0 and
                              len(state.inventories[opponentId].ants) == 1))):
            return True
        else:
            return False

    ##
    # canAttack
    #
    # Definition:   Determines whether or not a given ant can attack an enemy ant
    #
    # Parameters:
    #   state - the state at the given time
    #   coord - The coordinate of the ant in question
    #
    # Return: If there is no ant there, return False. Otherwise, true or false if an enemy ant is adjacent or not
    def canAttack(self, state, coord):
        # If there is no ant there, return False
        if getAntAt(state, coord) is None:
            return False

        # Determine whether an enemy ant is nearby
        adjacent = listAdjacent(coord)
        for spot in adjacent:
            ant = getAntAt(state, spot)
            if ant is not None and ant.player is not self.playerId:
                if spot[0] == ant.coords[0] or spot[1] == ant.coords[1]:
                    return True

        return False

        ##
        # canAttack
        #
        # Definition:   Determines whether or not a given ant can attack an enemy ant
        #
        # Parameters:
        #   myInv - the state at the given time
        #   state - currentState
        #
        # Return: If there is no ant there, return False. Otherwise, true

    def evalQueen(self, myInv, state, foodList):
        queen = myInv.getQueen()

        # We don't want our queen on top of food or the anthill
        if queen.coords == getConstrList(state, self.playerId, (ANTHILL,))[0].coords or queen.coords == foodList[0] or \
                        queen.coords == foodList[1]:
            return 0
        return 1

    def evalWorkers(self, myInv, state):
        workers = getAntList(state, self.playerId, (WORKER,))
        if len(workers) < 1 and len(workers) > 3:
            return 0
        elif len(workers) is 2:
            return 0.5
        return 1

    def evalWorkerCarrying(self, myInv, state):
        # Find worker ants not carrying
        CarryingWorkers = []
        for ant in myInv.ants:
            if ant.carrying and ant.type == WORKER:
                CarryingWorkers.append(ant)

        antDistScore = 0
        for ant in CarryingWorkers:
            minDist = None
            tunnelDist = 10000
            for tunnel in myInv.getTunnels():
                dist = self.dist(state, ant, tunnel.coords)
                if dist <= tunnelDist:
                    tunnelDist = dist
            antHillDist = self.dist(state, ant, myInv.getAnthill().coords)
            if tunnelDist <= antHillDist:
                minDist = tunnelDist
            else:
                minDist = antHillDist
            antDistScore += self.scoreDist(minDist, 14)
        if len(CarryingWorkers) > 0:
            score = antDistScore / float(len(CarryingWorkers))
        else:
            return 0

        return score

    def evalWorkerNotCarrying(self, myInv, state):
        # Find worker ants not carrying
        notCarryingWorkers = []
        for ant in myInv.ants:
            if (not ant.carrying) and ant.type == WORKER:
                notCarryingWorkers.append(ant)

        antDistScore = 0
        for ant in notCarryingWorkers:
            minDist = 1000
            foodList = []
            for constr in state.inventories[2].constrs:
                if constr.type == FOOD:
                    foodList.append(constr)

            for food in foodList:
                dist = self.dist(state, ant, food.coords)
                if dist <= minDist:
                    minDist = dist

            antDistScore += self.scoreDist(minDist, 14)

        if len(notCarryingWorkers) > 0:
            score = antDistScore / float(len(notCarryingWorkers))
        else:
            return 0

        return score

    def evalFoodCount(self, myInv, state):
        count = myInv.foodCount / 11
        return count

    def dist(self, gameState, ant, dest):
        diffX = abs(ant.coords[0] - dest[0])
        diffY = abs(ant.coords[1] - dest[1])
        return diffX + diffY

    def scoreDist(self, dist, bound):
        # score= dif/10 + .5 (for abs(dif) < 5 else dif is +-5)
        print "dist", dist
        if dist == 0:
            return 1.0
        if dist > bound:
            dist = bound
        return (-dist + bound) / float(bound)


# ## Unit Tests
#
# #unit test 1##
# p = AIPlayer(0)
# p.NUM_NODES = 4
# p.ALPHA = 0.8
# p.networkWeights = [0.5, 0.3, -0.3, 0.0, 0.9, 0.2, 0.0, 0.0, -0.4, -0.8, -0.4, 0.1, -0.1]
# testInputs =  [0,1]
# output = 0.444
# out = p.propagateNeuralNetwork(testInputs)
# diff =  out[p.NUM_NODES-1] - output
# if diff < 0.03 and diff > -0.03:
#     print "propagate test passed."
# else: print "propagate test failed."
#
# p.backPropagateNeuralNetwork(1.0, out, testInputs)
# string = "["
# for w in p.networkWeights:
#     string += " {0:.3f} ".format(w)
# print string, "]"
