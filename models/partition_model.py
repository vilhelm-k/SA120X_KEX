import gurobipy as gp
from gurobipy import GRB
from itertools import permutations, combinations
import numpy as np
import time
from .base_model import BaseModel

# Import the BaseModel class - this would actually be a relative import in practice


class PartitionModel(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Initialize the set partitioning model, extending the base model.
        """
        super().__init__(*args, **kwargs)

        # Additional data structures for route generation and set partitioning
        self.routes_pool = []  # List of route sequences [(start, task1, task2, ..., end), ...]
        self.route_costs = []  # Total duration of each route
        self.route_tasks = []  # List of tasks in each route [[task1, task2, ...], ...]
        self.route_caregivers = []  # Caregiver assigned to each route

        # Maps for efficient constraint generation
        self.task_routes = {}  # Maps task to routes covering it {task_id: [route_idx1, route_idx2, ...], ...}
        self.caregiver_routes = {}  # Maps caregiver to routes {caregiver_id: [route_idx1, route_idx2, ...], ...}

        # Solution variables
        self.lambda_vars = None  # Binary variables for route selection

    def build(self, max_tasks_per_route=5, caregiver_penalty=60):
        """
        Build the set partitioning model by:
        1. Generating feasible routes
        2. Creating the set partitioning constraints

        Args:
            max_tasks_per_route: Maximum number of tasks in a route
            caregiver_penalty: Penalty for using a caregiver

        Returns:
            The built Gurobi model
        """
        print("Building set partitioning model for home healthcare routing...")
        start_time = time.time()

        # Create the optimization model
        self.model = gp.Model("HomeCare_SetPartitioning")

        # Generate all feasible routes
        self._generate_routes(max_tasks_per_route)
        route_gen_time = time.time() - start_time
        print(f"Route generation complete in {route_gen_time:.2f} seconds")
        print(f"Generated {len(self.routes_pool)} feasible routes")

        # Create binary decision variables for routes
        self.lambda_vars = self.model.addVars(range(len(self.routes_pool)), vtype=GRB.BINARY, name="route")

        # Each task must be covered exactly once
        for i in self.V:
            if i in self.task_routes and self.task_routes[i]:
                self.model.addConstr(
                    gp.quicksum(self.lambda_vars[r] for r in self.task_routes[i]) == 1, f"Cover_task_{i}"
                )
            else:
                print(f"Warning: Task {i} has no feasible routes")

        # Each caregiver can be assigned at most one route
        for k in self.K:
            if k in self.caregiver_routes and self.caregiver_routes[k]:
                self.model.addConstr(
                    gp.quicksum(self.lambda_vars[r] for r in self.caregiver_routes[k]) <= 1, f"Caregiver_{k}_usage"
                )

        # Objective: minimize total time + caregiver penalty
        self.model.setObjective(
            gp.quicksum(self.route_costs[r] * self.lambda_vars[r] for r in range(len(self.routes_pool)))
            + caregiver_penalty
            * gp.quicksum(
                gp.quicksum(self.lambda_vars[r] for r in self.caregiver_routes[k])
                for k in self.K
                if k in self.caregiver_routes
            ),
            GRB.MINIMIZE,
        )

        build_time = time.time() - start_time
        print(f"Model built in {build_time:.2f} seconds")
        print(f"Model has {self.model.NumVars} variables and {self.model.NumConstrs} constraints")

        return self.model

    def _generate_routes(self, max_tasks_per_route=15, max_routes_per_caregiver=10000):
        """
        Generate all feasible routes for each caregiver.

        Args:
            max_tasks_per_route: Maximum number of tasks in a route
            max_routes_per_caregiver: Maximum number of routes to generate per caregiver
        """
        print(f"Generating routes with up to {max_tasks_per_route} tasks per route...")

        # Initialize route mapping structures
        self.task_routes = {i: [] for i in self.V}
        self.caregiver_routes = {k: [] for k in self.K}

        # Generate routes for each caregiver
        for k in self.K:
            qualified_tasks = self.caregiver_tasks[k]
            if not qualified_tasks:
                continue

            routes_generated = 0

            # Generate routes of different lengths
            for route_length in range(1, min(max_tasks_per_route + 1, len(qualified_tasks) + 1)):
                # For each possible combination of tasks of the given length
                for task_subset in combinations(qualified_tasks, route_length):
                    # For each possible ordering of tasks
                    for task_sequence in permutations(task_subset):
                        # Check if this sequence is temporally feasible
                        is_feasible, route_info = self._check_route_feasibility(k, task_sequence)

                        if is_feasible:
                            # Add route to the pool
                            route_idx = len(self.routes_pool)

                            # Store the route
                            self.routes_pool.append(("start",) + task_sequence + ("end",))
                            self.route_costs.append(route_info["duration"])
                            self.route_tasks.append(task_sequence)
                            self.route_caregivers.append(k)

                            # Update mappings
                            for task in task_sequence:
                                self.task_routes[task].append(route_idx)
                            self.caregiver_routes[k].append(route_idx)

                            routes_generated += 1

                        # Check if we've hit the limit for this caregiver
                        if routes_generated >= max_routes_per_caregiver:
                            break

                    if routes_generated >= max_routes_per_caregiver:
                        break

                if routes_generated >= max_routes_per_caregiver:
                    break

            print(f"Generated {routes_generated} feasible routes for caregiver {k}")

    def _check_route_feasibility(self, k, task_sequence):
        """
        Check if a route is temporally feasible for a given caregiver.

        Args:
            k: Caregiver ID
            task_sequence: Sequence of tasks [(task1, task2, ...)]

        Returns:
            (is_feasible, route_info): Tuple with feasibility flag and route information
        """
        current_location = "start"
        current_time = 0
        task_times = {}  # Start times for each task

        # Check each task in sequence
        for task in task_sequence:
            # Travel time from current location to task
            travel_time = self.c[k, current_location, task]

            # Arrival time at task
            arrival_time = current_time + travel_time

            # Check if arrival is before latest start time
            if arrival_time > self.l[task]:
                return False, {}

            # Service start time (exactly at earliest start time since we know arrival times are fixed)
            service_start = max(arrival_time, self.e[task])
            task_times[task] = service_start

            # Complete service
            service_end = service_start + self.s[task]

            # Update current state
            current_time = service_end
            current_location = task

        # Travel back to end depot
        travel_time = self.c[k, current_location, "end"]
        end_time = current_time + travel_time

        # Calculate total duration
        total_duration = end_time  # Since we start at time 0

        return True, {"duration": total_duration, "task_times": task_times}

    def _extract_arrival_times(self):
        """
        Extract arrival times from the solution.

        Returns:
            Dictionary mapping caregivers to arrival times at each location
        """
        self.arrivals = {}

        # Initialize arrival times dictionary
        for k in self.K:
            self.arrivals[k] = {}

        # Extract selected routes
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            for r in range(len(self.routes_pool)):
                if self.lambda_vars[r].X > 0.5:  # If route is selected
                    k = self.route_caregivers[r]
                    route_sequence = self.routes_pool[r]
                    task_sequence = self.route_tasks[r]

                    # Simulate the route execution to get arrival times
                    current_location = "start"
                    current_time = 0

                    # Record start time
                    self.arrivals[k]["start"] = current_time

                    # Process each task
                    for task in task_sequence:
                        # Travel time to task
                        travel_time = self.c[k, current_location, task]
                        arrival_time = current_time + travel_time

                        # Service start time
                        service_start = max(arrival_time, self.e[task])
                        self.arrivals[k][task] = service_start

                        # Update current state
                        current_time = service_start + self.s[task]
                        current_location = task

                    # Record end time
                    travel_time = self.c[k, current_location, "end"]
                    end_time = current_time + travel_time
                    self.arrivals[k]["end"] = end_time

        return self.arrivals
