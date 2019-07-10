

from ortools.graph import pywrapgraph

numNodes = 8
numArcs = 16
startNodes = [ 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
endNodes   = [ 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7]
capacities =  [0.06, 0.06, 0.06, 0.06, 0.06, 0.11, 0.11, 0.11, 0.06, 0.11, 0.17, 0.17, 0.06, 0.11, 0.17, 0.67]
unitCosts =  [ 0, 37, 77, 85, 39, 0, 38, 46, 77, 40, 0, 8, 85, 48, 8, 0]
supplies = [0.06, 0.11, 0.17, 0.67, -0.06, -0.11, -0.17, -0.67]

# Instantiate a SimpleMinCostFlow solver.
min_cost_flow = pywrapgraph.SimpleMinCostFlow()

# Add each arc.
for i in range(0, len(startNodes)):
    min_cost_flow.AddArcWithCapacityAndUnitCost(startNodes[i], endNodes[i],
                                                int(100 *capacities[i]), unitCosts[i])

# Add node supplies.
for i in range(0, len(supplies)):
    min_cost_flow.SetNodeSupply(i, int(100 * supplies[i]))

# Find the minimum cost flow 
if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
    print('Minimum cost:', min_cost_flow.OptimalCost() / 100)
    print('')
    print('  Arc    Flow / Capacity  Cost')
    for i in range(min_cost_flow.NumArcs()):
      cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
      print('%1s -> %1s   %3s  / %3s       %3s' % (
          min_cost_flow.Tail(i),
          min_cost_flow.Head(i),
          min_cost_flow.Flow(i),
          min_cost_flow.Capacity(i),
          cost))
else:
    print('There was an issue with the min cost flow input.')