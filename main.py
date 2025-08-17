import csv
from dataclasses import dataclass
import random
from typing import List, Tuple
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, LpStatusOptimal, lpSum, PULP_CBC_CMD


@dataclass
class Person:
    name: str
    gender: str
    weight: float
    side: str


def parse_csv(filepath: str) -> List[Person]:
    people = []
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            person = Person(
                name=row["name"],
                gender=row["gender"],
                weight=float(row["weight"]),
                side=row["side"]
            )
            people.append(person)
    return people


def build_boat(people: List[Person]) -> Tuple[List[Person], List[Person]]:
    """
    Use linear programming to select 20 people for a boat with optimal weight distribution.

    Constraints:
    - 20 people total
    - 10 males, 10 females
    - 10 righties, 10 lefties (ambidextrous can count as either)
    - Minimize weight difference between left and right sides of boat
    """
    n = len(people)

    # Create the optimization problem
    prob = LpProblem("Boat_Selection", LpMinimize)

    # Decision variables: x[i] = 1 if person i is selected, 0 otherwise
    x = [LpVariable(f"select_{i}", cat='Binary') for i in range(n)]

    # Decision variables: left[i] = 1 if person i goes to left side, 0 otherwise
    left = [LpVariable(f"left_{i}", cat='Binary') for i in range(n)]

    # Decision variables: right[i] = 1 if person i goes to right side, 0 otherwise
    right = [LpVariable(f"right_{i}", cat='Binary') for i in range(n)]

    # Variable for weight difference (to minimize)
    weight_diff = LpVariable("weight_diff", lowBound=0)

    # Constraint: Select exactly 20 people
    prob += lpSum(x) == 20

    # Constraint: 10 males
    prob += lpSum(x[i] for i in range(n) if people[i].gender == 'M') == 10

    # Constraint: 10 females
    prob += lpSum(x[i] for i in range(n) if people[i].gender == 'F') == 10

    # Constraint: 10 righties (R or A)
    prob += lpSum(x[i] for i in range(n) if people[i].side in ['R', 'A']) >= 10

    # Constraint: 10 lefties (L or A)
    prob += lpSum(x[i] for i in range(n) if people[i].side in ['L', 'A']) >= 10

    # Constraint: Each selected person must be assigned to exactly one side
    for i in range(n):
        prob += left[i] + right[i] == x[i]

    # Constraint: Lefties (L) must go to left side, Righties (R) must go to right side
    # Ambidextrous (A) can go to either side
    for i in range(n):
        if people[i].side == 'L':
            prob += right[i] == 0  # Lefties cannot go to right side
        elif people[i].side == 'R':
            prob += left[i] == 0  # Righties cannot go to left side

    # Constraint: 10 people on each side of the boat
    prob += lpSum(left) == 10
    prob += lpSum(right) == 10

    # Calculate weight difference
    left_weight = lpSum(people[i].weight * left[i] for i in range(n))
    right_weight = lpSum(people[i].weight * right[i] for i in range(n))

    # Constraint: weight_diff >= |left_weight - right_weight|
    prob += weight_diff >= left_weight - right_weight
    prob += weight_diff >= right_weight - left_weight

    # Objective: Minimize weight difference
    prob += weight_diff

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output

    # Check if solution is optimal
    if prob.status != LpStatusOptimal:
        raise ValueError(f"No optimal solution found. Status: {LpStatus[prob.status]}")

    # Extract the solution
    left_people = []
    right_people = []

    for i in range(n):
        if x[i].varValue == 1:  # Person is selected
            if left[i].varValue == 1:
                left_people.append(people[i])
            else:
                right_people.append(people[i])

    return left_people, right_people


# Example usage:
if __name__ == "__main__":
    data = parse_csv("roster.csv")
    left, right = build_boat(data)
    print("Left side:")
    for p in left:
        print(p)
    print("\nRight side:")
    for p in right:
        print(p)

    print(f"\nWeight difference: {abs(sum(p.weight for p in left) - sum(p.weight for p in right)):.2f}")
    print(f"Left side weight: {sum(p.weight for p in left):.2f}")
    print(f"Right side weight: {sum(p.weight for p in right):.2f}")

    # Verify constraints
    left_males = sum(1 for p in left if p.gender == 'M')
    left_females = sum(1 for p in left if p.gender == 'F')
    right_males = sum(1 for p in right if p.gender == 'M')
    right_females = sum(1 for p in right if p.gender == 'F')

    all_people = left + right
    total_righties = sum(1 for p in all_people if p.side in ['R', 'A'])
    total_lefties = sum(1 for p in all_people if p.side in ['L', 'A'])

    print(f"\nConstraint verification:")
    print(f"Total people: {len(all_people)}")
    print(f"Males: {left_males + right_males}, Females: {left_females + right_females}")
    print(f"People who can be righties: {total_righties}")
    print(f"People who can be lefties: {total_lefties}")
    print(f"Left side: {len(left)}, Right side: {len(right)}")