from ortools.linear_solver import pywraplp

# Define dummy data
N = range(2)  # set of all power plants
K = range(2)  # set of all ship types
T = range(3)  # set of all days in the planning horizon
a = {0: 10, 1: 15}  # fuel consumption per unit distance traveled by ships of type k (ton/n mile)
b = 20  # unit fuel price (USD/ton)
c = {0: 100, 1: 150}  # time-charter cost of renting a ship of type k for |T| days (USD)
d = {0: 2, 1: 3}  # maximum number of ships of type k that can be chartered
l = {0: 50, 1: 70}  # total length of a trip from power plant i to the storage location 0 and then back to the power plant (n mile)
m = {0: 200, 1: 300}  # CO2 tank capacity of ships of type k (ton)
v = {0: 20, 1: 25}  # sailing speed of ships of type k (n mile/hour)
q = {0: 500, 1: 600}  # CO2 storage capacity of power plant i (ton)
r = 30  # benefit of transporting a ton of CO2 from power plants to the storage location compared to emitting a ton of CO2 into the atmosphere (USD/ton)
n = {(0, 0): 1, (0, 1): 2, (1, 0): 1, (1, 1): 2}  # sailing time of ships of type k to complete a trip from power plant i to the storage location 0 and then back to the power plant (day)
w = {(0, 0): 50, (0, 1): 60, (1, 0): 70, (1, 1): 80, (2, 0): 90, (2, 1): 100}  # amount of CO2 produced by power plant i in day t (ton)

# Create a solver
solver = pywraplp.Solver.CreateSolver('GLOP')

# Define decision variables
alpha = {}
epsilon = {}
beta = {}
gamma = {}
delta = {}

for i in N:
    for k in K:
        alpha[i, k] = solver.IntVar(0, solver.infinity(), 'alpha[%i,%i]' % (i, k))
        for t in T:
            epsilon[i, k, t] = solver.IntVar(0, solver.infinity(), 'epsilon[%i,%i,%i]' % (i, k, t))

for i in N:
    for t in T:
        beta[i, t] = solver.NumVar(0, solver.infinity(), 'beta[%i,%i]' % (i, t))
        gamma[i, t] = solver.NumVar(0, solver.infinity(), 'gamma[%i,%i]' % (i, t))
        delta[i, t] = solver.NumVar(0, q[i], 'delta[%i,%i]' % (i, t))

# Define objective function
objective = solver.Objective()

for i in N:
    for t in T:
        objective.SetCoefficient(gamma[i, t], r)  # Objective 1

for i in N:
    for k in K:
        objective.SetCoefficient(alpha[i, k], -c[k])  # Objective 2

for i in N:
    for k in K:
        for t in T:
            objective.SetCoefficient(epsilon[i, k, t], -b * a[k] * l[i])  # Objective 3

objective.SetMaximization()
# Constraint 1: Ensures that the total number of chartered ships of each type does not exceed the maximum allowed.
# Constraints 2 and 3: Ensure that the number of ships departing from each power plant does not exceed the number of chartered ships.
# Constraint 4: Sets the initial CO2 storage at each power plant to zero.
# Constraint 5: Updates the CO2 storage at each power plant based on emissions, loading, and unloading.
# Constraint 6: Limits the CO2 storage at each power plant to its capacity.
# Constraint 7: Ensures that the amount of CO2 transported by each ship type does not exceed the capacity of the ships.

# Add constraints
for k in K:
    solver.Add(sum(alpha[i, k] for i in N) <= d[k])  # Constraint 1

for i in N:
    for k in K:
        for t in T[1:]:
            solver.Add(sum(epsilon[i, k, t_prime] for t_prime in range(max(0, t - n[i, k]), t)) <= alpha[i, k])  # Constraint 2

for i in N:
    for k in K:
        for t in T[:-1]:
            solver.Add(sum(epsilon[i, k, t_prime] for t_prime in range(1, t - n[i, k] + 1)) <= alpha[i, k])  # Constraint 3

for i in N:
    for t in T:
        if t == 0:
            solver.Add(delta[i, t] == 0)  # Constraint 4
        else:
            solver.Add(delta[i, t] == delta[i, t - 1] + w[t,i] - beta[i, t] - gamma[i, t])  # Constraint 5

for i in N:
    solver.Add(delta[i, t] <= q[i])  # Constraint 6

for i in N:
    for t in T:
        solver.Add(gamma[i, t] <= sum(m[k] * epsilon[i, k, t] for k in K))  # Constraint 7

# Solve the problem
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    for i in N:
        for k in K:
            print('alpha[%i,%i] = %d' % (i, k, alpha[i, k].solution_value()))
            for t in T:
                print('epsilon[%i,%i,%i] = %d' % (i, k, t, epsilon[i, k, t].solution_value()))
    for i in N:
        for t in T:
            print('beta[%i,%i] = %f' % (i, t, beta[i, t].solution_value()))
            print('gamma[%i,%i] = %f' % (i, t, gamma[i, t].solution_value()))
            print('delta[%i,%i] = %f' % (i, t, delta[i, t].solution_value()))
else:
    print('The problem does not have an optimal solution.')
