# Dijkstra's Shortest Path Algorithm
## AIM

To develop a code to find the shortest route from the source to the destination point using Dijkstra's shortest path algorithm.

## THEORY
Best-first search algorithm always selects the path which appears best at that moment. It is the combination of depth-first search and breadth-first search algorithms. Best-first search allows us to take the advantages of both algorithms. With the help of best-first search, at each step, we can choose the most promising node. In the best first search algorithm, we expand the node which is closest to the goal node.
The best first search uses the concept of a priority queue. It is a search algorithm that works on a specific rule. The aim is to reach the goal from the initial state via the shortest path. Best First Search is an algorithm for finding the shortest path from a given starting node to a goal node in a graph. The algorithm works by expanding the nodes of the graph in order of increasing the distance from the starting node until the goal node is reached.


## DESIGN STEPS

### STEP 1:
Identify a location in the google map.

### STEP 2:
Select a specific number of nodes with distance.

### STEP 3:
Start from the initial node and put it in the ordered list.

### STEP 4:
Repeat the next steps until the GOAL node is reached.

## ROUTE MAP

#### Example map
![My map](https://user-images.githubusercontent.com/75234965/167985082-11ef27db-bbf7-4408-bfc1-d0af0125f841.PNG)


## PROGRAM
```
%matplotlib inline
import matplotlib.pyplot as plt
import random
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations
import heapq

class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds): 
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        
        raise NotImplementedError
    def result(self, state, action): 
        raise NotImplementedError
    def is_goal(self, state):        
        return state == self.goal
    def action_cost(self, s, a, s1): 
        return 1
    
    def __str__(self):
        return '{0}({1}, {2})'.format(
            type(self).__name__, self.initial, self.goal)
            
class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __str__(self): 
        return '<{0}>'.format(self.state)
    def __len__(self): 
        return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): 
        return self.path_cost < other.path_cost

failure = Node('failure', path_cost=math.inf) 
cutoff  = Node('cutoff',  path_cost=math.inf)

def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)
        

def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []  
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None): 
        return []
    return path_states(node.parent) + [node.state]

class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x): 
        self.key = key
        self.items = [] 
        for item in items:
            self.add(item)
         
    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]
    
    def top(self): return self.items[0][1]

    def __len__(self): return len(self.items)
    
def best_first_search(problem, f):
    "Search nodes with minimum f(node) value first."
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return failure

def g(n): 
    return n.path_cost
    cost = 1
    return cost

class RouteProblem(Problem):
    """A problem to find a route between locations on a `Map`.
    Create a problem with RouteProblem(start, goal, map=Map(...)}).
    States are the vertexes in the Map graph; actions are destination states."""
    
    def actions(self, state): 
        """The places neighboring `state`."""
        return self.map.neighbors[state]
    
    def result(self, state, action):
        """Go to the `action` place, if the map says that is possible."""
        return action if action in self.map.neighbors[state] else state
    
    def action_cost(self, s, action, s1):
        """The distance (cost) to go from s to s1."""
        return self.map.distances[s, s1]
    
    def h(self, node):
        "Straight-line distance between state and the goal."
        locs = self.map.locations
        return straight_line_distance(locs[node.state], locs[self.goal])

class Map:
    """A map of places in a 2D world: a graph with vertexes and links between them. 
    In `Map(links, locations)`, `links` can be either [(v1, v2)...] pairs, 
    or a {(v1, v2): distance...} dict. Optional `locations` can be {v1: (x, y)} 
    If `directed=False` then for every (v1, v2) link, we add a (v2, v1) link."""

    def __init__(self, links, locations=None, directed=False):
        if not hasattr(links, 'items'): # Distances are 1 by default
            links = {link: 1 for link in links}
        if not directed:
            for (v1, v2) in list(links):
                links[v2, v1] = links[v1, v2]
        self.distances = links
        self.neighbors = multimap(links)
        self.locations = locations or defaultdict(lambda: (0, 0))

        
def multimap(pairs) -> dict:
    "Given (key, val) pairs, make a dict of {key: [val,...]}."
    result = defaultdict(list)
    for key, val in pairs:
        result[key].append(val)
    return result

Mapping_locations = Map(
    {('Vadapalani', 'Kodambakkam'): 2,('Kodambakkam', 'T.Nagar'): 2,('T.Nagar', 'Gopalpuram'): 4,('Gopalpuram', 'Marina'): 3,('Marina', 'ALwarpet'): 5,('Alwarpet', 'MRC Nagar'): 4,('Alwarpet', 'Saidapet'): 6,('Saidepet', 'Kasi Theatre'): 3,('Kasi Theatre', 'MGR Nagar'): 2,('MGR Nagar', 'Valasaravakkam'): 4,
    ('Valasaravakkam', 'Vadapalani'): 5,('Valasaravakkam', 'Mugalivakkam'): 3,('Mugalivakkam', 'Kolapakkam'): 2,('Kolapakkam', 'Airport'): 8,('Airport', 'Ramapuram'): 9,('Ramapuram', 'Nandambakkam'): 18,('Nandambakkam', 'MGR Nagar'): 20,('Ramapuram', 'Guindy'): 8,('Guindy', 'Little Mount'): 2,
    ('Little Mount', 'Saidapet'): 3,('Guindy', 'Indian Institute of Technology'): 8,('Indian Institute of Technology', 'Velachery'): 9,('T.Nagar', 'Alwarpet'): 3, ('T.Nagar', 'MGR Nagar'): 6})
    
 r0 = RouteProblem('Vadapalani', 'Guindy', map=Mapping_locations)
r1 = RouteProblem('Kasi Theatre', 'Vadapalani', map=Mapping_locations)
r2 = RouteProblem('Velachery', 'Guindy', map=Mapping_locations)
r3 = RouteProblem('Marina', 'MRC Nagar', map=Mapping_locations)
r4 = RouteProblem('Ramapuram', 'MGR Nagar', map=Mapping_locations)
print(r0)
print(r1)
print(r2)
print(r3)
print(r4)

goal_state_path=best_first_search(r0,g)
path_states(goal_state_path) 
print("GoalStateWithPath:{0}".format(goal_state_path))
print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))

goal_state_path=best_first_search(r1,g)
path_states(goal_state_path) 
print("GoalStateWithPath:{0}".format(goal_state_path))
print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


goal_state_path=best_first_search(r2,g)
path_states(goal_state_path) 
print("GoalStateWithPath:{0}".format(goal_state_path))
print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


goal_state_path=best_first_search(r3,g)
path_states(goal_state_path) 
print("GoalStateWithPath:{0}".format(goal_state_path))
print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))

goal_state_path=best_first_search(r4,g)
path_states(goal_state_path) 
print("GoalStateWithPath:{0}".format(goal_state_path))
print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))

```


## OUTPUT:

![3 1](https://user-images.githubusercontent.com/75234965/167985314-eada4115-a3cc-4fa4-a469-4b7ffa4a1ee7.PNG)

![3 2](https://user-images.githubusercontent.com/75234965/167985318-c82139e9-e11e-4db5-a5d2-e5c18f27917e.PNG)

![3 3](https://user-images.githubusercontent.com/75234965/167985326-10d63649-6f21-4ba3-9d92-2a111dbbf165.PNG)

![3 4](https://user-images.githubusercontent.com/75234965/167985351-a88f47ce-139e-4688-938f-44d1508ed90e.PNG)

![3 5](https://user-images.githubusercontent.com/75234965/167985357-5fa16f12-c451-4281-84e1-eb29ee14b58c.PNG)

## Justification:
n best-first search algorithm, the selected node is verified as parent node or not and starts its search, within the least distance it will be reaching the goal node. Search near every two nodes are always considered with its shortest distance.

## RESULT:
Thus an algorithm to find the route from the source to the destination point using best-first search is developed and executed successfully.




