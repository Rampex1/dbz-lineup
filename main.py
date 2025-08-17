import csv
from dataclasses import dataclass
import random
from typing import List, Tuple, Dict
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, LpStatusOptimal, lpSum, PULP_CBC_CMD


@dataclass
class Person:
    name: str
    gender: str
    weight: float
    side: str


@dataclass
class Boat:
    size: int
    gender: str  # "Open" (only males), "Women" (only females), "Mixed" (half male, half female)


def parse_csv(filepath: str) -> List[Person]:
    people = []
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            person = Person(
                name=row["Name"],
                gender=row["Gender"],
                side=row["Side"],
                weight = float(row["Weight"])
            )
            people.append(person)
    return people


def build_boat(people: List[Person], boats: List[Boat]) -> Dict[int, Tuple[List[Person], List[Person]]]:
    """
    Use linear programming to assign people to multiple boats with optimal weight distribution.

    Args:
        people: List of available people
        boats: List of boats with size and gender requirements

    Returns:
        Dictionary mapping boat index to (left_side, right_side) tuple

    Constraints:
    - Each boat must be filled to exactly its size
    - Open boats: only males
    - Women boats: only females
    - Mixed boats: exactly half male, half female
    - Each person on a boat: lefties go left, righties go right, ambidextrous can go either side
    - Each boat has equal numbers on left and right sides
    - Minimize the maximum weight difference across all boats
    """
    n_people = len(people)
    n_boats = len(boats)

    # Validate boat sizes are even (needed for left/right split)
    for i, boat in enumerate(boats):
        if boat.size % 2 != 0:
            raise ValueError(f"Boat {i} has odd size {boat.size}. All boats must have even sizes for left/right split.")

    # Create the optimization problem
    prob = LpProblem("Multi_Boat_Selection", LpMinimize)

    # Decision variables: assign[person][boat][side] = 1 if person is assigned to boat on given side
    assign = {}
    for p in range(n_people):
        assign[p] = {}
        for b in range(n_boats):
            assign[p][b] = {
                'left': LpVariable(f"assign_{p}_{b}_left", cat='Binary'),
                'right': LpVariable(f"assign_{p}_{b}_right", cat='Binary')
            }

    # Variable for maximum weight difference (to minimize)
    max_weight_diff = LpVariable("max_weight_diff", lowBound=0)

    # Constraint: Each person can be assigned to at most one boat and one side
    for p in range(n_people):
        total_assignments = []
        for b in range(n_boats):
            total_assignments.extend([assign[p][b]['left'], assign[p][b]['right']])
        prob += lpSum(total_assignments) <= 1

    # Constraint: Each boat must be filled to exactly its size
    for b in range(n_boats):
        boat_total = []
        for p in range(n_people):
            boat_total.extend([assign[p][b]['left'], assign[p][b]['right']])
        prob += lpSum(boat_total) == boats[b].size

    # Constraint: Equal numbers on left and right sides of each boat
    for b in range(n_boats):
        left_count = lpSum(assign[p][b]['left'] for p in range(n_people))
        right_count = lpSum(assign[p][b]['right'] for p in range(n_people))
        prob += left_count == boats[b].size // 2
        prob += right_count == boats[b].size // 2

    # Gender constraints for each boat
    for b in range(n_boats):
        boat = boats[b]

        if boat.gender == "Open":  # Only males
            for p in range(n_people):
                if people[p].gender == 'F':
                    prob += assign[p][b]['left'] == 0
                    prob += assign[p][b]['right'] == 0

        elif boat.gender == "Women":  # Only females
            for p in range(n_people):
                if people[p].gender == 'M':
                    prob += assign[p][b]['left'] == 0
                    prob += assign[p][b]['right'] == 0

        elif boat.gender == "Mixed":  # Half male, half female
            male_count = lpSum(assign[p][b]['left'] + assign[p][b]['right']
                               for p in range(n_people) if people[p].gender == 'M')
            female_count = lpSum(assign[p][b]['left'] + assign[p][b]['right']
                                 for p in range(n_people) if people[p].gender == 'F')
            prob += male_count == boat.size // 2
            prob += female_count == boat.size // 2

    # Handedness constraints: L goes left, R goes right, A can go either
    for p in range(n_people):
        for b in range(n_boats):
            if people[p].side == 'L':
                prob += assign[p][b]['right'] == 0  # Lefties cannot go to right side
            elif people[p].side == 'R':
                prob += assign[p][b]['left'] == 0  # Righties cannot go to left side

    # Calculate weight differences for each boat and constrain max_weight_diff
    for b in range(n_boats):
        left_weight = lpSum(people[p].weight * assign[p][b]['left'] for p in range(n_people))
        right_weight = lpSum(people[p].weight * assign[p][b]['right'] for p in range(n_people))

        # max_weight_diff >= |left_weight - right_weight| for this boat
        prob += max_weight_diff >= left_weight - right_weight
        prob += max_weight_diff >= right_weight - left_weight

    # Objective: Minimize maximum weight difference across all boats
    prob += max_weight_diff

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output

    # Check if solution is optimal
    if prob.status != LpStatusOptimal:
        raise ValueError(f"No optimal solution found. Status: {LpStatus[prob.status]}")

    # Extract the solution
    result = {}
    for b in range(n_boats):
        left_people = []
        right_people = []

        for p in range(n_people):
            if assign[p][b]['left'].varValue == 1:
                left_people.append(people[p])
            elif assign[p][b]['right'].varValue == 1:
                right_people.append(people[p])

        result[b] = (left_people, right_people)

    return result

# Example usage:
if __name__ == "__main__":
    data = parse_csv("roster.csv")

    # Example boats
    boats = [
        Boat(size=20, gender="Mixed"),  # 20 people, 10 male, 10 female
        Boat(size=16, gender="Open"),  # 16 people, all male
        Boat(size=12, gender="Women"),  # 12 people, all female
    ]

    boat_assignments = build_boat(data, boats)

    for boat_idx, (left, right) in boat_assignments.items():
        print(f"\n=== BOAT {boat_idx + 1} ({boats[boat_idx].gender}, size {boats[boat_idx].size}) ===")
        print("Left side:")
        for p in left:
            print(f"  {p}")
        print("\nRight side:")
        for p in right:
            print(f"  {p}")

        left_weight = sum(p.weight for p in left)
        right_weight = sum(p.weight for p in right)
        weight_diff = abs(left_weight - right_weight)

        print(f"\nWeight difference: {weight_diff:.2f}")
        print(f"Left side weight: {left_weight:.2f}")
        print(f"Right side weight: {right_weight:.2f}")

        # Verify constraints
        left_males = sum(1 for p in left if p.gender == 'M')
        left_females = sum(1 for p in left if p.gender == 'F')
        right_males = sum(1 for p in right if p.gender == 'M')
        right_females = sum(1 for p in right if p.gender == 'F')

        total_males = left_males + right_males
        total_females = left_females + right_females

        print(f"Males: {total_males}, Females: {total_females}")

        # Check handedness compliance
        wrong_side = []
        for p in left:
            if p.side == 'R':
                wrong_side.append(f"{p.name} (R on left)")
        for p in right:
            if p.side == 'L':
                wrong_side.append(f"{p.name} (L on right)")

        if wrong_side:
            print(f"⚠️  Handedness violations: {', '.join(wrong_side)}")
        else:
            print("✅ Handedness constraints satisfied")