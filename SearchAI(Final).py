__author__ = "Jacob Kirby and Danh Nguyen"
#CS421
#Neural Network

import random
import sys
#import numpy as numpy #for our matrix manipulations

sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
import time



##
# AIPlayer
# Description: The responsbility of this class is to interact with the game by
# deciding a valid move based on a given game state. This class has methods that
# will be implemented by students in Dr. Nuxoll's AI course.

#
# Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):
    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "WUBBA LUBBA DUB DUB")
        self.DESIRED_WORKERS = 2
        self.DESIRED_SOLDIERS = 2
        self.MIN_FOOD = 2
        self.depth = 2
        self.WINNING_VALUE = 1000
        self.LOSING_VALUE = 0

        #input 0
        self.tunnelDist = 0

        #input 1
        self.hillDist = 0

        #input 2
        self.totalWorkers = 0

        #input 3
        self.totalFood = 0


        #NEURAL NETWORK code
        self.weightList = []

        #biases for each node
        self.bias = 1

        #outputs for each node
        self.outputs = [0.0, 0.0, 0.0]


    ##
    # getPlacement
    #
    # Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        # implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]

    ##
    # getMove
    # Description: Gets the next move from the Player.
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##
    def getMove(self, currentState):
        move = self.expandCurrentState(currentState, 0)


        if move == None:
            move = Move(END, None, None)

        return move

    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    #----------------------------------------------------------------------------------------------------------------#
                                            #HW5: NEURAL NETS#
    #----------------------------------------------------------------------------------------------------------------#
    ##
    # mapState
    #
    # Description: Takes GameState and evaluates it with a neural net
    #
    def mapState(self, state):
        #create array
        stateMapping = []
        foodList = []
        self.myFood = None
        self.myTunnel = None
        self.myHill = None

        # the first time this method is called, the food and tunnel locations
        # need to be recorded in their respective instance variables
        if self.myTunnel is None and len(getConstrList(prevState, self.playerId, (TUNNEL,))) > 0:
            self.myTunnel = getConstrList(prevState, self.playerId, (TUNNEL,))[0]
        if self.myHill is None:
            self.myHill = getConstrList(prevState, self.playerId, (ANTHILL,))[0]
        if self.myFood is None:
            foods = getConstrList(prevState, None, (FOOD,))
            self.myFood = foods[0]
            # find the food closest to the tunnel
            bestDistSoFar = 1000  # i.e., infinity
            for food in foods:
                if food.coords[1] < 4:
                    foodList.append(food)


        #store references of players' resources
        myInv = state.inventories[state.whoseTurn]

        foeInv = state.inventories[(state.whoseTurn+1) % 2]

        myQueen = myInv.getQueen()

        myWorkers = getAntList(state, state.whoseTurn, (WORKER,))[0]

        foeQueen = foeInv.getQueen()

        targetFood = foodList[0]



        #store references of ant distances
        # Find the closest food to this ant
        bestDistSoFar = 1000
        for worker in myWorkers:
            for food in foodList:
                if approxDist(workers[worker].coords, food.coords) < bestDistSoFar:
                    bestDistSoFar = approxDist(workers[worker].coords, food.coords)
                    targetFood = food


        # Find the closest tunnel to this ant
        if approxDist(workers[worker].coords, self.myTunnel.coords) < approxDist(workers[worker].coords, self.myHill.coords):
            targetTunnel = self.myTunnel
        else:
            targetTunnel = self.myHill


        # queen on hill -> 0 or 1
        stateMapping.append(float(myQueen.coords == self.hillCoords))

        stateMapping.append((len(myInv.ants) - 1))


        #Default ant distances are 1
        stateMapping.append(0.0) #worker 1 from hill
        stateMapping.append(0.0) #worker 1  from food
        stateMapping.append(0.0) #worker 2 from hill
        stateMapping.append(0.0) #worker 2  from food
        stateMapping.append(0.0) #worker 3 from hill
        stateMapping.append(0.0) #worker 3  from food
        stateMapping.append(0.0) #soldier 1 from hill
        stateMapping.append(0.0) #soldier 1

        # if player has lost, set the "win/loss" value to 0.0
        if playerQueen is None or enemyInv.foodCount >= 11:
            stateArray.append(0.0)
        # if player has won, set the "win/loss" value to 1.0
        elif enemyQueen is None or playerInv.foodCount >= 11:
            stateArray.append(1.0)
        # haven't won or loss yet, keep "win/loss" value at 0.5
        else:
            stateArray.append(0.5)


        antDistanceIdx = 2
        for ant in myInv.ants:
            if ant != myQueen:
                stateMapping[antDistanceIdx] = approxDist(ant.coords, targetTunnel)
                antDistanceIdx += 1
                stateMapping[antDistanceIdx] = approxDist(ant.coords, targetFood)
                antDistanceIdx += 1
                if (antDistanceIdx >= 8):
                    break


        stateMapping.append(self.bias) #bias starts at 1

        return stateMapping


    ##
    # initNeuralNet
    #
    # Description: Takes GameState and evaluates it with a neural net
    #
    def initNeuralNet(self):
        return 0


    def matrixOps(self, stateMapping):
        return 0

    def applyGFxn(self, matrix1, matrix2):
        resultMatrix = np.(matrix1, matrix2) #use numpy to multiply both matrices togeher
        return [[1/(1 + math.e**-el) for el in resultMatrix[0]]]




    ##
    # neuralNetEval
    #
    # Description: Takes GameState and evaluates it with a neural net
    #
    def neuralNetEval(self, currentState):
        # Default value is 0.5, we will add/subtract from here
        value = 0.5

        # Grab the playerIDs
        oppId = 0 if self.playerId == 1 else 1
        foodList = []
        self.myFood = None
        self.myTunnel = None
        self.myHill = None

        ##### OUR STUFF #####
        inventory = getCurrPlayerInventory(nextState)
        workers = getAntList(nextState, self.playerId, (WORKER,))
        soldiers = getAntList(nextState, self.playerId, (SOLDIER,))
        drone = getAntList(nextState, self.playerId, (DRONE,))
        ranger = getAntList(nextState, self.playerId, (R_SOLDIER,))
        queen = inventory.getQueen()

        prevInventory = getCurrPlayerInventory(prevState)
        prevWorkers = getAntList(prevState, self.playerId, (WORKER,))
        prevSoldiers = getAntList(prevState, self.playerId, (SOLDIER,))
        prevQueen = prevInventory.getQueen()

        prevOppFighters = getAntList(prevState, oppId, (SOLDIER, DRONE, R_SOLDIER,))

        # the first time this method is called, the food and tunnel locations
        # need to be recorded in their respective instance variables
        if self.myTunnel is None and len(getConstrList(prevState, self.playerId, (TUNNEL,))) > 0:
            self.myTunnel = getConstrList(prevState, self.playerId, (TUNNEL,))[0]
        if self.myHill is None:
            self.myHill = getConstrList(prevState, self.playerId, (ANTHILL,))[0]
        if self.myFood is None:
            foods = getConstrList(prevState, None, (FOOD,))
            self.myFood = foods[0]
            # find the food closest to the tunnel
            bestDistSoFar = 1000  # i.e., infinity
            for food in foods:
                if food.coords[1] < 4:
                    foodList.append(food)

        # Array to store inputs
        net_inputs = []




        # Winning/losing conditions
        if self.hasWon(nextState, oppId):
            value = self.LOSING_VALUE
            return value
        elif self.hasWon(nextState, self.playerId):
            value = self.WINNING_VALUE
            return value



        ###
        # We need our workers to be moving towards some food.
        ###
        numWorkers = 0

        # if len(prevWorkers) > len(workers):
        #     numWorkers = len(workers)
        # else:
        #     numWorkers = len(prevWorkers)

        for idx in range(0, numWorkers):

            targetFood = foodList[0]
            targetTunnel = self.myTunnel

            # Find the closest food to this ant
            bestDistSoFar = 1000
            for food in foodList:
                if approxDist(prevWorkers[idx].coords, food.coords) < bestDistSoFar:
                    bestDistSoFar = approxDist(prevWorkers[idx].coords, food.coords)
                    targetFood = food


            # Find the closest tunnel to this ant
            if approxDist(prevWorkers[idx].coords, self.myTunnel.coords) < approxDist(prevWorkers[idx].coords, self.myHill.coords):
                targetTunnel = self.myTunnel
            else:
                targetTunnel = self.myHill

            # Compare how far we were to how close we are now, reward based on relative distance
            prevDistTunnel = approxDist(prevWorkers[idx].coords, targetTunnel.coords)
            #nextDistTunnel = approxDist(workers[idx].coords, targetTunnel.coords)
            prevDistFood = approxDist(prevWorkers[idx].coords, targetFood.coords)
            #nextDistFood = approxDist(workers[idx].coords, targetFood.coords)

            if (prevWorkers[idx].carrying or prevDistFood == 0) and prevDistTunnel != 0:
                if(approxDist(prevWorkers[idx].coords, targetTunnel.coords) > 7):
                    self.tunnelDist = 0.01
                elif((approxDist(prevWorkers[idx].coords, targetTunnel.coords) > 3)):
                    self.tunnelDist = 0.03
                elif((approxDist(prevWorkers[idx].coords, targetTunnel.coords) > 1)):
                    self.tunnelDist = 0.05
                elif((approxDist(prevWorkers[idx].coords, targetTunnel.coords) == 0)):
                    self.tunnelDist = 0.1

                #amount = self.rateMoveOnDist(prevDistTunnel, nextDistTunnel)
                #value = value + amount
            else:
                if(approxDist(prevWorkers[idx].coords, targetFood.coords) > 7):
                        value = value + 0.01
                elif((approxDist(prevWorkers[idx].coords, targetFood.coords) > 3)):
                    value = value + 0.03
                elif((approxDist(prevWorkers[idx].coords, targetFood.coords) > 1)):
                    value = value + 0.05
                elif((approxDist(prevWorkers[idx].coords, targetFood.coords) == 0)):
                    value = value + 0.1

        if len(self.myFood) > 1:
            value = value + 0.01
        elif len(self.myFood) > 3:
            value = value + 0.05
        elif len(self.myFood) > 8:
            value = value + 0.2



        # amount = self.rateMoveOnDist(prevDistFood, nextDistFood)
        # value = value + amount


        #Ant number preferences
        # Limit the number of ants we have for each type
        if len(workers) > self.DESIRED_WORKERS:
            value = value - 0.5
        elif len(workers) < self.DESIRED_WORKERS:
            value = value - 0.1
        else:
            value = value + 0.1

        if len(soldiers) > self.DESIRED_SOLDIERS:
            value = value - 0.0
        else:
            value = value + 0.0

        if len(drone) > 0 or len(ranger) > 0:
            value = value - 1


        prevApproxSoldierDist = 0
        nextOppWorker = getAntList(nextState, oppId, (WORKER,)) if (len(getAntList(nextState, oppId, (WORKER,)))
                                                                    > 0) else 0
        prevOppWorker = getAntList(prevState, oppId, (WORKER,)) if (len(getAntList(prevState, oppId, (WORKER,)))
                                                                    > 0) else 0
        numSoldiers = 0




    ##
    # rateState
    #
    # Description: Examines the given state and returns a decimal number
    #               depending on how "good" the state is
    #
    # STRATEGY #
    ##############
    # 1) always have workers collecting food
    # 2) Build soldiers
    # 3) Attack opponents workers
    # 4) Attack opponents queen
    #############
    #
    ##
    def rateState(self, prevState, nextState):



        ###
        # We need our soldiers to be moving towards a target. These targets are, in priority:
        #   1) Enemy Workers
        #   2) Enemy Queen
        ###

        # If we lost a soldier for some reason, don't try to include it
        if len(prevSoldiers) > len(soldiers):
            numSoldiers = len(soldiers)
        else:
            numSoldiers = len(prevSoldiers)

        if prevOppWorker != 0 and nextOppWorker != 0:
            # They have workers, so go towards them
            amount = 0
            for idx in range(0, numSoldiers):

                # find the closest enemy worker for this soldier
                closestDist = 1000
                closestWorker = (0, 0)
                numAttacking = 0
                for enemyWorker in prevOppWorker:
                    if approxDist(prevSoldiers[idx].coords, enemyWorker.coords) < closestDist:
                        closestDist = approxDist(prevSoldiers[idx].coords, enemyWorker.coords)
                        closestWorker = enemyWorker.coords

                # Reward ourselves if we made a move that was closer to an enemy worker
                prevDistSoldier = approxDist(prevSoldiers[idx].coords, closestWorker)
                nextDistSoldier = approxDist(soldiers[idx].coords, closestWorker)

                amount += self.rateMoveOnDist(prevDistSoldier, nextDistSoldier)

                # Reward ourselves if we can attack them
                if self.canAttack(nextState, soldiers[idx].coords):
                    numAttacking = numAttacking + 1

                if approxDist(soldiers[idx].coords, nextOppWorker[0].coords) == 1 or len(nextOppWorker) == len(
                        prevOppWorker) - 1:
                    value = value + .01

                amount += 0.1 * float(numAttacking) / numSoldiers

            value = value + amount

        else:
            # They don't have any workers, so go after their queen
            if len(soldiers) > 0 and len(prevSoldiers) > 0:

                amount = 0
                numAttacking = 0
                for idx in range(0, numSoldiers):

                    prevDistSoldier = approxDist(prevSoldiers[idx].coords, getAntList(nextState, oppId, (QUEEN,))[0].coords)
                    nextDistSoldier = approxDist(soldiers[idx].coords, getAntList(prevState, oppId, (QUEEN,))[0].coords)

                    # Reward ourselves if the move allows us to attack the queen
                    if self.canAttack(nextState, soldiers[idx].coords):
                        numAttacking = numAttacking + 1
                    else:
                        amount += self.rateMoveOnDist(prevDistSoldier, nextDistSoldier)

                amount += 0.2 * float(numAttacking) / numSoldiers

                value = value + amount


        # We don't want our queen on top of food or the anthill
        if queen.coords == getConstrList(nextState, self.playerId, (ANTHILL,))[0].coords or queen.coords == foodList[0] or \
                                                                               queen.coords == foodList[1]:
            value = value - .3

        # Edge case: Correct the value if it is below 0
        if value < 0:
            value = .001

        return value

        ##
        # evalNodeList
        #
        # Description: Evaluates a node list and rates all of them
        #
        # Parameters:
        #   nodes - The list of node objects to evaluate
        #
        # Returns: The average distance from 0.5 for each node
        ##
        def evalNodeList(self, nodes):
            if len(nodes) == 0:
                return 0

            totalVal = 0
            for node in nodes:
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
            begTimeList = time.clock()
            for move in moveList:
                if move.moveType != END:
                    antType = -1
                    if getAntAt(state, move.coordList[0]) is not None:
                        antType = getAntAt(state, move.coordList[0]).type

                    # We want to check if the queen is on our anthill
                    onAntHill = False
                    if move.coordList[len(move.coordList) - 1] == getConstrList(state,self.playerId,(ANTHILL,))[0].coords:
                        onAntHill = True

                    # We want to check if our queen is on some food
                    queenOnFood = False
                    for food in getConstrList(state, None, (FOOD,)):
                        if move.coordList[len(move.coordList) - 1] == food.coords:
                            queenOnFood = True
                            break

                    # We don't want to expand certain nodes, I'll let you read this to figure out which ones
                    if (move.moveType == MOVE_ANT and antType != QUEEN and len(move.coordList) > 1) or move.moveType == BUILD or \
                            (antType == QUEEN and not (onAntHill or queenOnFood) and move.moveType == MOVE_ANT and len(move.coordList) > 1) or \
                            ((antType == QUEEN or antType == SOLDIER) and self.canAttack(state, move.coordList[0])):

                        nextState = getNextState(state, move)
                        nodeList.append(Node(move, nextState, self.rateState(state, nextState), state))

            # EDGE CASE: One of the ants can't move, so it bricks the system. Force an ant to move.
            if len(nodeList) == 0 and len(moveList) != 0:
                for desparateMove in moveList:
                    nextState = getNextState(state, desparateMove)
                    nodeList.append(Node(desparateMove, nextState, self.rateState(state, nextState), state))


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
        # backpropogation
        # Description: Given a target value and actual value, determines
        #   how "wrong" the input weights of the perceptrons in the
        #   neural network are and adjusts them using the backpropogation
        #   algorithm. The higher the "alpha value", the larger the
        #   adjustments and vice versa. (See pseudocode in function's
        #   comments for the algorithm.)
        #
        # Notes:
        #   The neural network's weight matrices are changed by this
        #   function.
        #
        # Parameters:
        #   self - The object pointer
        #   targetVal - The expected value of the state (determined
        #       using evaluateState)
        #   actualVal - The actual value of the state (found
        #       using the neural network)
        #
        # Return: Nothing
        #
        def backpropogation(self, targetVal, actualVal):
           # determine output perceptron error
            err = targetVal - actualVal
            # calculate the delta value using the error
            delta = actualVal*(1-actualVal)*err

            # hidden layer perceptron errors
            errs = [weight[0]*delta for weight in self.outputLayerWeights]
            # calculate the deltas for all of the hidden layer perceptrons
            # using their respective errors
            deltas = [b*(1-b)*errs[idx] for idx, b in enumerate(self.hiddenLayerOutputs[0])]

            ##
            # Backpropogation algorithm!
            # Adjust each weight in the network:  W_ij = W_ij + alpha * delta_j * x_i where:
            # W_ij is the weight between nodes i and j
            # alpha is a learning rate
            # delta_j is the error term for node j
            # x_i  is the input that the weight was applied to
            #
            # adjust output layer weights
            self.outputLayerWeights = [[weight[0] + self.alpha*delta*self.hiddenLayerOutputs[0][idx]] for idx, weight in enumerate(self.outputLayerWeights)]

            # adjust hidden layer weights
            for idx_i, row in enumerate(self.hiddenLayerWeights):
                for idx_j, weight in enumerate(row):
                    self.hiddenLayerWeights[idx_i][idx_j] += self.alpha*deltas[idx_j]*self.neuralNetInput[idx_i]

            return






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
            if getAntAt(state,coord) is None:
                return False

            # Determine whether an enemy ant is nearby
            adjacent = listAdjacent(coord)
            for spot in adjacent:
                ant = getAntAt(state, spot)
                if ant is not None and ant.player is not self.playerId:
                    if spot[0] == ant.coords[0] or spot[1] == ant.coords[1]:
                        return True

            return False

#####
# Node
#
# Description:
#   Node defines a node object which contains a state, a value, a parent node, and a move.
#
#   Move - The move that will take us to the state
#   State - The state that would be reached if the move was taken
#   Parent Node - The node that the move was taken from
#   Value - The rating for the state, based on our rate node method above
class Node:
    def __init__(self, theMove, theState, theValue, theParent):
        self.move = theMove
        self.state = theState
        self.value = theValue
        self.parent = theParent
