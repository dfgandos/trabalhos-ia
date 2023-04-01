# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


# implementação realizado para disciplina de IA da PUC Minas
def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first."""

    # inicializações

    # "visited" contém nós que foram retirados da pilha,
    # e a direção de onde foram obtidos
    visited = {}

    # "solution" contém a sequência de direções para o Pacman chegar ao estado objetivo
    solution = []

    # "stack" contém trigêmeos de: (nó na lista de franjas, direção, custo)
    stack = util.Stack()

    # "parents" contém nós e seus pais
    parents = {}

    # o estado inicial é obtido e adicionado à pilha
    start = problem.getStartState()
    stack.push((start, 'Undefined', 0))

    # a direção de onde chegamos ao estado inicial é indefinida
    visited[start] = 'Undefined'

    # retorna se o próprio estado inicial for o objetivo
    if problem.isGoalState(start):
        return solution

    # loop enquanto a pilha não está vazia e o objetivo não é alcançado
    goal = False;
    while(stack.isEmpty() != True and goal != True):

        # pop do topo da pilha
        node = stack.pop()

        # armazena o elemento e sua direção
        visited[node[0]] = node[1]
        # verifica se o elemento é objetivo
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break
        # expande o nó
        for elem in problem.getSuccessors(node[0]):

            # se o sucessor ainda não foi visitado
            if elem[0] not in visited.keys():
                # loja sucessora e seu pai
                parents[elem[0]] = node[0]

                # empurra o sucessor para a pilha
                stack.push(elem)

    # encontrando e armazenando o caminho
    while(node_sol in parents.keys()):
        # encontra pai
        node_sol_prev = parents[node_sol]

        # prefixa a direção para a solução
        solution.insert(0, visited[node_sol])

        # vai para o nó anterior
        node_sol = node_sol_prev

    return solution

# implementação realizado para disciplina de IA da PUC Minas
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # inicializações

    # "visited" contém nós que foram retirados da fila,
    # e a direção de onde foram obtidos
    visited = {}

    # "solution" contém a sequência de direções para o Pacman chegar ao estado objetivo
    solution = []

    # "fila" contém trigêmeos de: (nó na lista de franjas, direção, custo)
    queue = util.Queue()

    # "parents" contém nós e seus pais
    parents = {}

    # estado inicial é obtido e adicionado à fila
    start = problem.getStartState()
    queue.push((start, 'Undefined', 0))

    # a direção de onde chegamos ao estado inicial é indefinida
    visited[start] = 'Undefined'

    # retorna se o próprio estado inicial for o objetivo
    if problem.isGoalState(start):
        return solution

    # loop enquanto a fila não está vazia e o objetivo não é alcançado
    goal = False;
    while(queue.isEmpty() != True and goal != True):

        # pop do topo da fila
        node = queue.pop()

        # armazena o elemento e sua direção
        visited[node[0]] = node[1]
        # verifica se o elemento é objetivo
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break

        # expande o nó
        for elem in problem.getSuccessors(node[0]):

            # se o sucessor ainda não foi visitado ou expandido como filho de outro nó
            if elem[0] not in visited.keys() and elem[0] not in parents.keys():
                # loja sucessora e seu pai
                parents[elem[0]] = node[0]

                # coloca o sucessor na fila
                queue.push(elem)

    # encontrando e armazenando o caminho
    while(node_sol in parents.keys()):
        # encontra pai
        node_sol_prev = parents[node_sol]

        # prefixa a direção para a solução
        solution.insert(0, visited[node_sol])

        # vai para o nó anterior
        node_sol = node_sol_prev

    return solution

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    # inicializações

    # "visited" contém nós que foram retirados da fila,
    # e a direção de onde foram obtidos
    visited = {}

    # "solution" contém a sequência de direções para o Pacman chegar ao estado objetivo
    solution = []

    # "fila" contém trigêmeos de: (nós na lista de franjas, direção, custo)
    queue = util.PriorityQueue()

    # "parents" contém nós e seus pais
    parents = {}

    # "cost" contém nós e seus custos correspondentes
    cost = {}

    # estado inicial é obtido e adicionado à fila
    start = problem.getStartState()
    queue.push((start, 'Undefined', 0), 0)

    # a direção de onde chegamos ao estado inicial é indefinida
    visited[start] = 'Undefined'
    # custo do estado inicial é 0
    cost[start] = 0

    # retorna se o próprio estado inicial for o objetivo
    if problem.isGoalState(start):
        return solution

    # loop enquanto a fila não está vazia e o objetivo não é alcançado
    goal = False;
    while(queue.isEmpty() != True and goal != True):

        # pop do topo da fila
        node = queue.pop()

        # armazena o elemento e sua direção
        visited[node[0]] = node[1]

        # verifica se o elemento é objetivo
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break

        # expande o nó
        for elem in problem.getSuccessors(node[0]):

            # se o sucessor não for visitado, calcule seu novo custo
            if elem[0] not in visited.keys():
                priority = node[2] + elem[2]

                # se o custo do sucessor foi calculado anteriormente ao expandir um nó diferente,
                # se o novo custo for maior que o antigo, continue
                if elem[0] in cost.keys():
                    if cost[elem[0]] <= priority:
                        continue

                # se o novo custo for menor que o custo antigo, coloque na fila e altere o custo e o pai
                queue.push((elem[0], elem[1], priority), priority)
                cost[elem[0]] = priority
                # loja sucessora e seu pai
                parents[elem[0]] = node[0]

    # encontrando e armazenando o caminho
    while(node_sol in parents.keys()):

        # encontra pai
        node_sol_prev = parents[node_sol]

        # prefixa a direção para a solução
        solution.insert(0, visited[node_sol])

        # vai para o nó anterior
        node_sol = node_sol_prev

    return solution

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # inicializações

    # "visited" contém nós que foram retirados da fila,
    # e a direção de onde foram obtidos
    visited = {}

    # "solution" contém a sequência de direções para o Pacman chegar ao estado objetivo
    solution = []

    # "fila" contém trigêmeos de: (nó na lista de franjas, direção, custo)
    queue = util.PriorityQueue()

    # "parents" contém nós e seus pais
    parents = {}

    # "cost" contém nós e seus custos correspondentes
    cost = {}

    # estado inicial é obtido e adicionado à fila
    start = problem.getStartState()
    queue.push((start, 'Undefined', 0), 0)

    # a direção de onde chegamos ao estado inicial é indefinida
    visited[start] = 'Undefined'

    # custo do estado inicial é 0
    cost[start] = 0

    # retorna se o próprio estado inicial for o objetivo
    if problem.isGoalState(start):
        return solution

    # loop enquanto a fila não está vazia e o objetivo não é alcançado
    goal = False;
    while(queue.isEmpty() != True and goal != True):

        # pop do topo da fila
        node = queue.pop()

        # armazena o elemento e sua direção
        visited[node[0]] = node[1]

        # verifica se o elemento é objetivo
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break

        # expande o nó
        for elem in problem.getSuccessors(node[0]):

            # se o sucessor não for visitado, calcule seu novo custo
            if elem[0] not in visited.keys():
                priority = node[2] + elem[2] + heuristic(elem[0], problem)

                # se o custo do sucessor foi calculado anteriormente ao expandir um nó diferente,
                # se o novo custo for maior que o antigo, continue
                if elem[0] in cost.keys():
                    if cost[elem[0]] <= priority:
                        continue

                # se o novo custo for menor que o custo antigo, coloque na fila e altere o custo e o pai
                queue.push((elem[0], elem[1], node[2] + elem[2]), priority)
                cost[elem[0]] = priority

                # loja sucessora e seu pai
                parents[elem[0]] = node[0]

    # encontrando e armazenando o caminho
    while(node_sol in parents.keys()):

        # encontra pai
        node_sol_prev = parents[node_sol]

        # prefixa a direção para a solução
        solution.insert(0, visited[node_sol])

        # vai para o nó anterior
        node_sol = node_sol_prev

    return solution

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch