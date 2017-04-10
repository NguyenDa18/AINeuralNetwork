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
from operator import itemgetter, attrgetter
from random import shuffle



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

        super(AIPlayer,self).__init__(inputPlayerId, "Testing")
        self.weHaveNotDoneThisBefore = True
        self.enemyFood = []
        self.ourFood = []

        # ############# NEURAL NETWORK CODE #############
        # self.weightList = [0.9940292021713424, 1.28358055070786586, -0.8853893842112309, 0.33343, 1.9574090852542456,
        #                    0.9850658493134115, 0.11991592831956542, 0.1819871994251512,- 0.1, 1.5940292021713384,
        #                    1.3835805507078652, 1.9853893842112309, -.2,.2, .1,
        #                   .1, .1, .2, .15, .12,
        #                    .1, 3, 3, 3, 3, 1.857409085254275, 1.44039591616289453, 1.101809608671836, 0.3]

        self.weightList = [0.5, 0.2, 0.5, -0.2, -0.8, 0.9, 0.1 , 0.1, 0.1, 0.1, 0.3, 0.6, 0.7 , 0.8, 0.4,
        -0.5, -0.1, 0.2, 0.5, 0.6, 0.3, 0.1, 0.2, 0.9, -0.7, 0.1, 0.1, 0.1, 0.3]
        self.ALPHA = .7

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
        # get food lists
        if self.weHaveNotDoneThisBefore:
            foods = getConstrList(currentState, None, (FOOD,))
            for food in foods:
                if food.coords[1] > 3:
                    self.enemyFood.append(food)
                else:
                    self.ourFood.append(food)
            self.weHaveNotDoneThisBefore = False

        return (self.moveSearch(2, 0, self.initNode(None, currentState, True, None))['move'])

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

    # #
    # initNode
    # Description: Create a new Node and return it
    #
    # Parameters:
    #   move - the move to create the next node
    #   currentState - a clone of the current state
    ##
    def initNode(self, move, currentState, isMaxNode, parentNode):
        if move is None:
            nextState = currentState
        else:
            nextState = getNextStateAdversarial(currentState, move)

        if isMaxNode:
            bound = 0
        else:
            bound = 1

        node = {'move': move, 'currState': currentState, 'nextState': nextState,
                'utility': self.neuralNetEval(currentState, nextState),
                'isMax': isMaxNode,
                'bound': bound, 'parentNode': parentNode}
        return node

    # #
    # evalNode                                          DEPRECIATED
    # Description: Takes a dictionary of node and returns the average utility
    #
    # Parameters:
    #   nodes - a dictionary list of nodes to be evaluated
    ##
    def evalNode(self, nodes):
        util = 0
        for node in nodes:
            util += node['utility']

        return float(util) / float(len(nodes))

    ##
    # moveSearch                                <!-- RECURSIVE -->
    #
    # Description: Takes the game state, depth, and a node and expands the node
    # using the current state. It then picks the node with the best utility and then
    # repeats this process until the desired depth has been reached.
    #
    # Parameters:
    #   state - the current game state
    #   depth - the depth we are currently at
    #   currNode - the node we are expanding
    #
    # Return:
    #   list of the moves to reach the most desireable state, list[-2] is the
    #   first move that can be taken
    ##
    def moveSearch(self, finalDepth, currDepth, currNode):
        if currDepth >= finalDepth or currDepth >= 5:
            currNode['utility'] = self.neuralNetEval(currNode['currState'], currNode['nextState'])
            return currNode

        # get list of neighboring nodes
        nodes = []
        for move in listAllLegalMoves(currNode['nextState']):
            if move.moveType == END:
                nodes.append(self.initNode(move, currNode['nextState'], not currNode['isMax'], currNode))
            else:
                nodes.append(self.initNode(move, currNode['nextState'], currNode['isMax'], currNode))

        # shuffle(nodes)
        if currNode['isMax']:
            nodes = sorted(nodes, key=itemgetter('utility'), reverse=True)[0:7]
        else:
            nodes = sorted(nodes, key=itemgetter('utility'), reverse=False)[0:7]

        for node in nodes:
            node = self.moveSearch(finalDepth, currDepth + 1, node)
            if currDepth != 0:
                if currNode['isMax'] and node['utility'] > currNode['bound']:
                    currNode['bound'] = node['utility']
                    if currNode['isMax'] is not currNode['parentNode']['isMax']:
                        if currNode['bound'] <= currNode['parentNode']['bound']:
                            currNode['bound'] = currNode['utility']
                            return currNode
                elif not currNode['isMax'] and node['utility'] < currNode['bound']:
                    currNode['bound'] = node['utility']
                    if currNode['isMax'] is not currNode['parentNode']['isMax']:
                        if currNode['bound'] >= currNode['parentNode']['bound']:
                            currNode['bound'] = currNode['utility']
                            return currNode

        if currNode['isMax']:
            maxUtil = -1
            for node in nodes:
                if node['utility'] > maxUtil:
                    maxUtil = node['utility']
                    favNode = node
            currNode['utility'] = maxUtil
        else:
            minUtil = 2
            for node in nodes:
                if node['utility'] < minUtil:
                    minUtil = node['utility']
                    favNode = node
            currNode['utility'] = minUtil

        if currDepth == 0:
            return favNode
        else:
            return currNode

    ##
    # hasWon
    # Description: This function was copied from Game.py so we could use it here.
    # Parameters:
    #   state -the current state
    #   playerId -current player id
    #
    # Return: boolean value for if one side has won
    #
    def hasWon(self, currentState, playerId):
        opponentId = (playerId + 1) % 2

        if ((currentState.phase == PLAY_PHASE) and
                ((currentState.inventories[opponentId].getQueen() == None) or
                     (currentState.inventories[opponentId].getAnthill().captureHealth <= 0) or
                     (currentState.inventories[playerId].foodCount >= FOOD_GOAL) or
                     (currentState.inventories[opponentId].foodCount == 0 and
                              len(currentState.inventories[opponentId].ants) == 1))):
            return True
        else:
            return False


    # ----------------------------------------------------------------------------------------------------------------#
                            # HW5: NEURAL NETS#
    # ----------------------------------------------------------------------------------------------------------------#
    # #
    # gfx
    #
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

    def neuralNetEval(self, currentState, nextState):

        # Grab the playerIDs
        oppId = 0 if self.playerId == 1 else 1
        #
        foodList = []
        self.myFood = None

        # ##### OUR STUFF #####
        inventory = getCurrPlayerInventory(currentState)
        enemyInv = currentState.inventories[oppId]


        if self.checkIfWon(inventory, enemyInv):
            return 1.0
        elif self.checkIfLose(inventory, enemyInv):
            return 0.0

        # assigns closest foods
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
        #evaluation methods
        net_inputs.append(self.evalQueen(inventory, currentState, foodList))
        net_inputs.append(self.evalWorkers(inventory, enemyInv, currentState))
        net_inputs.append(self.evalWorkerCarrying(inventory, currentState, nextState))
        net_inputs.append(self.evalWorkerNotCarrying(inventory, currentState, nextState))
        net_inputs.append(self.evalFoodCount(inventory, currentState))

        print net_inputs , "net inputs"

        #for the return statementm the average of all the scores
        inputSum = 0.0
        for input in net_inputs:
            inputSum += input

        #we want out output to be 1, which means we have won the game
        target = 1.0

        # call forward and back propogation
        net_outputs = self.propagateNeuralNetwork(net_inputs)
        self.backPropagateNeuralNetwork(target, net_outputs, net_inputs)

        error = net_outputs[numNetNodes - 1] - target
        print "Error: \n"
        print error
        print "weight List: ",self.weightList

        return inputSum

    # #
    # rateMoveOnDist
    # Description: Helper function for evalWorkerCarrying; given two distances, return score
    #
    # Parameters:
    #   dist1 -current location
    #   dist2 -destination location
    #
    # Return: Score of ditance. Value ignored and turned to zero if it is less than zero.
    # #
    def rateMoveOnDist(self, dist1, dist2):
        amount = .1 * ((float(dist1) - dist2) / dist1)
        return amount

    def checkIfWon(self, ourInv, enemyInv):
        if enemyInv.getQueen() is None or ourInv.foodCount == 11:
            return True
        return False

    # #
    # CheckIfLose
    # Description: Checks if the game state is a lose condition
    #
    # Parameters:
    #   ourInv - the AI's inventory
    #   enemyInv - the opponent's Inventory
    #
    # Return: Boolean: True if win condition, False if not.
    # #
    def checkIfLose(self, ourInv, enemyInv):
        # bit more complicated....
        if ourInv.getQueen() is None:
            return True
        return False

    ##
    # propagationNeuralNetwork
    # Description: Propogates through the neuaral network and outputs an score for the output node.
    #
    # Parameters:
    #   inputs -our list of input scores determined by our eval function
    #
    # Return: The value of each node in the network
    #
    def propagateNeuralNetwork(self, inputs):
        # constant # for how many nodes we have in network
        numNetNodes = 5
        nodeValues = [0] * numNetNodes
        # output node will always be last, but subtracted one so when we do loops, it will be iterated correctly
        output = numNetNodes - 1
        count = 0

        # bias generation = 4 biases for the inputs and 1 for output
        for count in range(0, numNetNodes):
            nodeValues[count] = self.weightList[count]
            count += 1

        # 20 weights from inputs to hidden nodes
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

        return nodeValues

    ##
    # backPropagateNeuralNetwork
    # Description: Adjust the weights of the network based on a target output.
    #
    # Parameters:
    #   target -our intended score
    #   outputs -list of outputs from our nodes, last element is our network output
    #   inputs -list of inputs to our neural network
    #
    # Return: Overall output of the network
    #
    def backPropagateNeuralNetwork(self, target, outputs, inputs):
        numNetNodes = 5
        # the amount of error on output node
        error = target - outputs[numNetNodes - 1]
        # counter used to find hidden nodes error
        counter = (numNetNodes - 1) + (numNetNodes - 1) * len(inputs)

        # delta used for back propagation equations
        delta = outputs[numNetNodes - 1] * (1 - outputs[numNetNodes - 1]) * error

        hiddenErrors = [0] * (numNetNodes - 1)
        hiddenDeltas = [0] * (numNetNodes - 1)

        # assigns each node with a error and delta
        for i in range(numNetNodes - 2):
            hiddenErrors[i] = self.weightList[counter + 1 + i] * delta
            hiddenDeltas[i] = outputs[i] * (1 - outputs[i]) * hiddenErrors[i]

        hiddenDeltas.append(delta)

        #each weight is changed using the backprop equation
        for weight in range(len(self.weightList) - 1):
            # for bias weights
            if weight < numNetNodes:
                nodeIndex = weight % numNetNodes
                # bias is 1
                input = 1
            # for the wieghts of the hidden nodes to output
            elif weight > len(self.weightList) - numNetNodes:
                nodeIndex = numNetNodes - 1
                inputIdx = weight - (len(self.weightList) - numNetNodes)
                input = inputs[inputIdx]
            # for input to hidden node weights
            else:
                nodeIndex = (weight - 1) % (numNetNodes - 1)
                inputIdx = (weight - numNetNodes) / (numNetNodes - 1)
                input = inputs[inputIdx]

            # backprop equation
            self.weightList[weight] += self.ALPHA * hiddenDeltas[nodeIndex] * input

    ##
    # hasWon
    # Description: This function was copied from Game.py so we could use it here.
    # Parameters:
    #   state -the current state
    #   playerId -current player id
    #
    # Return: boolean value for if one side has won
    #
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
    # evalQueen
    # Description: Evaluates score of queen's position, to be stored as input for our neural net
    # Parameters:
    #   state -the current state
    #   playerId -my player id
    #   foodList -list of food on the anthill
    #
    # Return: eval score for the queen position
    #
    def evalQueen(self, myInv, state, foodList):
        queen = myInv.getQueen()
        # We don't want our queen on top of food or the anthill
        if queen.coords == getConstrList(state, self.playerId, (ANTHILL,))[0].coords or queen.coords == foodList[0] or \
                        queen.coords == foodList[1]:
            return 0
        return 1

    ##
    # evalWorkers
    # Description: Evaluates score of our worker quantity, to be used as input for our neural network.
    # Parameters:
    #   state -the current state
    #
    # Return: eval score for worker
    #
    def evalWorkers(self, myInv, enemyInv, state):
        workerCount = 0
        droneCount = 0
        for ant in myInv.ants:
            if ant.type == SOLDIER:
                return 0
            if ant.type == R_SOLDIER:
                return 0
            if ant.type == WORKER:
                workerCount += 1
            if ant.type == DRONE:
                return 0

        if workerCount <= 1:
            return 0
        elif workerCount >= 3:
            return 0
        return 1

    def diff(self, ours, theirs, bound):
        # score= dif/10 + .5 (for abs(dif) < 5 else dif is +-5)
        diff = ours - theirs
        if diff >= bound:
            diff = bound
        elif diff <= bound:
            diff = -bound

        # return score
        return diff / (bound * 2) + 0.5

    ##
    # evalWorkerCarrying
    # Description: Evaluates score of our food gatherer worker, to be used as input for our neural network.
    # Parameters:
    #   myInv -my current inventory
    #   state -the current state
    #
    # Return: eval score of our food collecting worker
    #
    def evalWorkerCarrying(self, myInv, prevState, nextState):
        foodList = []
        value = 0
        self.myFood = None
        self.myTunnel = None
        self.myHill = None
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

        workers = getAntList(nextState, self.playerId, (WORKER,))
        prevWorkers = getAntList(prevState, self.playerId, (WORKER,))
        if len(prevWorkers) > len(workers):
            numWorkers = len(workers)
        else:
            numWorkers = len(prevWorkers)

        for idx in range(0, numWorkers):

            targetFood = foodList[0]
            targetTunnel = self.myTunnel

            # Find the closest food to this ant
            bestDistSoFar = 1000
            for food in foodList:
                if approxDist(workers[idx].coords, food.coords) < bestDistSoFar:
                    bestDistSoFar = approxDist(workers[idx].coords, food.coords)
                    targetFood = food

            # Find the closest tunnel to this ant
            if approxDist(workers[idx].coords, self.myTunnel.coords) < approxDist(workers[idx].coords,
                                                                                  self.myHill.coords):
                targetTunnel = self.myTunnel
            else:
                targetTunnel = self.myHill

            # Compare how far we were to how close we are now, reward based on relative distance
            prevDistTunnel = approxDist(prevWorkers[idx].coords, targetTunnel.coords)
            nextDistTunnel = approxDist(workers[idx].coords, targetTunnel.coords)
            prevDistFood = approxDist(prevWorkers[idx].coords, targetFood.coords)
            nextDistFood = approxDist(workers[idx].coords, targetFood.coords)

            if (prevWorkers[idx].carrying or prevDistFood == 0) and prevDistTunnel != 0:
                amount = self.rateMoveOnDist(prevDistTunnel, nextDistTunnel)
                if amount < 0:
                    amount = 0
                value = float(value) + amount
        print value
        return value/2


    ##
    # evalWorkerNotCarrying
    # Description: Evaluates score of worker not collecting food, so that it does.
    # Parameters:
    #   state -the current state
    #   myInv -my current inventory
    #
    # Return: eval score of workers not collecting food
    #
    def evalWorkerNotCarrying(self, myInv, prevState, nextState):
        foodList = []
        value = 0
        self.myFood = None
        self.myTunnel = None
        self.myHill = None
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

        workers = getAntList(nextState, self.playerId, (WORKER,))
        prevWorkers = getAntList(prevState, self.playerId, (WORKER,))

        if len(prevWorkers) > len(workers):
            numWorkers = len(workers)
        else:
            numWorkers = len(prevWorkers)

        for idx in range(0, numWorkers):

            targetFood = foodList[0]
            targetTunnel = self.myTunnel

            # Find the closest food to this ant
            bestDistSoFar = 1000
            for food in foodList:
                if approxDist(workers[idx].coords, food.coords) < bestDistSoFar:
                    bestDistSoFar = approxDist(workers[idx].coords, food.coords)
                    targetFood = food

            # Find the closest tunnel to this ant
            if approxDist(workers[idx].coords, self.myTunnel.coords) < approxDist(workers[idx].coords,
                                                                                  self.myHill.coords):
                targetTunnel = self.myTunnel
            else:
                targetTunnel = self.myHill

            # Compare how far we were to how close we are now, reward based on relative distance
            prevDistTunnel = approxDist(prevWorkers[idx].coords, targetTunnel.coords)
            nextDistTunnel = approxDist(workers[idx].coords, targetTunnel.coords)
            prevDistFood = approxDist(prevWorkers[idx].coords, targetFood.coords)
            nextDistFood = approxDist(workers[idx].coords, targetFood.coords)

            if (prevWorkers[idx].carrying or prevDistFood == 0) and prevDistTunnel != 0:
                continue
            else:
                amount = self.rateMoveOnDist(prevDistFood, nextDistFood)
                if amount < 0:
                    amount = 0
                value = float(value) + amount
        print value
        return value/2

    ##
    # evalFoodCount
    # Description: Evaluates score of our food amount, to be used as input for our neural network.
    # Parameters:
    #   myInv -my current inventory
    #   state -the current state
    #
    # Return eval score for our food count.
    #
    def evalFoodCount(self, myInv, state):
        print myInv.foodCount , "Food COunt "
        return float(myInv.foodCount)/11

    ##
    # dist
    # Description: Calculates the distance between an ant and coordinate
    # Parameters:
    #   gameState - the state of the game.
    #   ant - the ant
    #   dest - the destination coord
    #
    # Return: Score - based on difference of overall ant distance
    #
    def dist(self, gameState, ant, dest):
        diffX = abs(ant.coords[0] - dest[0])
        diffY = abs(ant.coords[1] - dest[1])
        return diffX + diffY

    ##
    # scoreDist
    # Description: Helper method to provide a score for distance based scores.
    # Parameters:
    #   dist -distance between two locations
    #   bound -maximum possible distance
    #
    # Return: Score - based on the distance and uses the bound to normalize number to be between 0 and 1.
    #
    def scoreDist(self, dist, bound):
        # based on a difference and a bound, calcualte a score
        if dist == 0:
            return 1.0
        if dist > bound:
            dist = bound
        return (-dist + bound) / float(bound)
