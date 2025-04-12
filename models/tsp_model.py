import gurobipy as gp
from gurobipy import GRB
from .base_model import BaseModel
import itertools


class TSPModel(BaseModel):
    def build(
        self,
        overtime_penalty=1,
        caregiver_penalty=60,
        worktime_per_break=5 * 60,
        regular_hours=8 * 60,
        break_length=30,
        continuity_penalty=45,
        day_continuity_penalty=10,
        lateness_penalty=5,
    ):
        print("Building TSP model.")
        # ---- Base Model Construction ----
        self.model = gp.Model("HomeCare")

        # Precompute all variable indices and weights
        # Create variable indices for x
        x_indices = []
        w_values = {}  # Store weights for each (k,i,j) combination

        for k in self.K:
            # Start-to-task routes
            for i in self.V:
                x_indices.append((k, "start", i))
                w_values[(k, "start", i)] = self.c(k, "start", i) + self.s[i]

                # Task-to-end routes
                x_indices.append((k, i, "end"))
                w_values[(k, i, "end")] = self.c(k, i, "end")

                # Task-to-task routes
                for j in self.V:
                    if i != j:
                        x_indices.append((k, i, j))
                        w_values[(k, i, j)] = (
                            self.l[j]
                            - self.l[i]
                            - lateness_penalty * (min(0, self.e[j] - self.l[i] - self.c(k, i, j)))
                        )

        # Create variables in batches
        print("Creating batch variables...")

        # Create x variables (route variables)
        x_vars = self.model.addVars(x_indices, vtype=GRB.BINARY, name="x")
        self.x = x_vars

        # Create caregiver usage variables
        is_used_vars = self.model.addVars(self.K, vtype=GRB.BINARY, name="is_used")
        self.is_used = is_used_vars

        # Create total time variables
        T_vars = self.model.addVars(self.K, vtype=GRB.CONTINUOUS, lb=0, name="T")
        self.T = T_vars

        # Store weights for each route
        self.w = w_values

        print("Created base variables.")

        # ---- Base Constraints ----
        print("Adding constraints in batches...")

        # Define caregiver usage constraints
        usage_constrs = {}
        for k in self.K:
            usage_constrs[k] = self.is_used[k] == gp.quicksum(self.x[k, "start", i] for i in self.V)
        self.model.addConstrs((usage_constrs[k] for k in self.K), name="Used")

        # Each task is visited exactly once by exactly one caregiver
        visit_constrs = {}
        for i in self.V:
            visit_constrs[i] = gp.quicksum(self.x[k, j, i] for k in self.K for j in self.V + ["start"] if j != i) == 1
        self.model.addConstrs((visit_constrs[i] for i in self.V), name="UniqueVisit")

        # Flow conservation for regular tasks
        flow_constrs = {}
        for k in self.K:
            for i in self.V:
                flow_constrs[k, i] = (
                    gp.quicksum(self.x[k, i, j] for j in self.V + ["end"] if j != i)
                    - gp.quicksum(self.x[k, j, i] for j in self.V + ["start"] if j != i)
                    == 0
                )
        self.model.addConstrs((flow_constrs[k, i] for k in self.K for i in self.V), name="Flow")

        # Only visit clients that the caregiver is qualified to visit
        qual_constrs = {}
        unqualified_visits = []
        for k in self.K:
            for j in self.V:
                if not self.is_caregiver_qualified(k, j):
                    for i in self.V + ["start"]:
                        if i != j and (k, i, j) in self.x:
                            unqualified_visits.append(self.x[k, i, j])

        if unqualified_visits:
            self.model.addConstr(
                gp.quicksum(unqualified_visits) == 0,
                name="Qualification",
            )

        # Define total time for each caregiver
        time_constrs = {}
        for k in self.K:
            time_constrs[k] = self.T[k] >= gp.quicksum(
                self.w[k, i, j] * self.x[k, i, j] for i in self.V for j in self.V if i != j and (k, i, j) in self.x
            ) + gp.quicksum(
                self.w[k, "start", i] * self.x[k, "start", i] + self.w[k, i, "end"] * self.x[k, i, "end"]
                for i in self.V
            )
        self.model.addConstrs((time_constrs[k] for k in self.K), name="TotalTime")

        # Base objective: minimize total time
        base_objective = gp.quicksum(self.T[k] for k in self.K)
        self.model.setObjective(base_objective, GRB.MINIMIZE)

        print("Built base model.")

        # ---- Optional Components ----
        objective_terms = [base_objective]

        # 1. Add overtime penalties if needed
        if overtime_penalty > 0:
            print("Adding overtime penalties.")
            # Create overtime variables in batch
            overtime_vars = self.model.addVars(self.K, vtype=GRB.CONTINUOUS, lb=0, name="overtime")
            self.overtime = overtime_vars

            # Add overtime constraints in batch
            overtime_constrs = {}
            for k in self.K:
                overtime_constrs[k] = self.overtime[k] >= self.T[k] - regular_hours
            self.model.addConstrs((overtime_constrs[k] for k in self.K), name="OvertimeCalculation")

            # Add overtime term to objective
            objective_terms.append(overtime_penalty * gp.quicksum(self.overtime[k] for k in self.K))

        # 2. Add caregiver penalty if needed
        if caregiver_penalty > 0:
            print("Adding caregiver usage penalties.")
            # Add caregiver penalty term to objective
            objective_terms.append(caregiver_penalty * gp.quicksum(self.is_used[k] for k in self.K))

        # 3. Add break requirements if needed
        if worktime_per_break > 0:
            print("Adding break requirements with evening shift exemption.")

            # Create evening shift variables in batch
            is_evening_shift_vars = self.model.addVars(self.K, vtype=GRB.BINARY, name="is_evening_shift")
            self.is_evening_shift = is_evening_shift_vars

            # Evening shift time threshold
            evening_shift_time = 15 * 60 + 30  # 15:30 in minutes since midnight

            # Set evening shift constraints in batch
            evening_constrs = {}
            for k in self.K:
                start_time = gp.quicksum(self.x[k, "start", i] * (self.e[i] - self.c(k, "start", i)) for i in self.V)
                evening_constrs[k] = self.is_evening_shift[k] <= (start_time - evening_shift_time) / 1440 + 1
            self.model.addConstrs((evening_constrs[k] for k in self.K), name="EveningShiftUpper")

            # Create break variables in batch
            B_vars = self.model.addVars([(k, i) for k in self.K for i in self.V], vtype=GRB.BINARY, name="B")
            self.B = B_vars

            # Add break constraints based on evening shift exemption
            for k in self.K:
                # One break for every worktime_per_break minutes, but only if not on evening shift
                self.model.addGenConstrIndicator(
                    self.is_evening_shift[k],
                    False,
                    gp.quicksum(self.B[k, i] for i in self.V) >= (self.T[k] / worktime_per_break - 1),
                    name=f"BreakRequired[{k}]",
                )

                # Can only take a break if caregiver visited a task with enough break time after
                break_constrs = {}
                for i in self.V:
                    # Collect all valid break pairs
                    valid_break_terms = []
                    for j in self.V:
                        if i != j and self.is_break_feasible(k, i, j, break_length) and (k, i, j) in self.x:
                            valid_break_terms.append(self.x[k, i, j])

                    if valid_break_terms:
                        break_constrs[k, i] = self.B[k, i] <= gp.quicksum(valid_break_terms)
                    else:
                        break_constrs[k, i] = self.B[k, i] == 0

                # Add all break constraints in batch
                self.model.addConstrs((break_constrs[k, i] for i in self.V), name=f"BreakVisit[{k}]")

        # 4. Add continuity of care penalties if needed
        if continuity_penalty > 0:
            print("Adding continuity of care penalties.")

            # Create new_serve variables in batch
            new_serve_vars = self.model.addVars(
                [(k, c) for k in self.K for c in self.C], vtype=GRB.BINARY, name="new_serve"
            )
            self.new_serve = new_serve_vars

            # Add constraints to detect new assignments
            serve_constrs = {}
            for k in self.K:
                for c in self.C:
                    client_tasks = self.get_client_tasks(c)
                    for j in client_tasks:
                        serve_constrs[k, c, j] = self.new_serve[k, c] >= gp.quicksum(
                            self.x[k, i, j] for i in self.V + ["start"] if i != j and (k, i, j) in self.x
                        )

            # Add all serve constraints in batch if there are any
            if serve_constrs:
                self.model.addConstrs(
                    (serve_constrs[k, c, j] for k in self.K for c in self.C for j in self.get_client_tasks(c)),
                    name="NewServe",
                )

            # Add continuity penalty term to objective
            objective_terms.append(
                gp.quicksum(
                    self.new_serve[k, c]
                    * (day_continuity_penalty + continuity_penalty * (1 - self.is_historically_visited(k, c)))
                    for k in self.K
                    for c in self.C
                )
            )

        # Update objective function with all terms
        if len(objective_terms) > 1:  # If we added more terms beyond the base
            self.model.setObjective(sum(objective_terms), GRB.MINIMIZE)
            print("Updated objective function with penalties.")

        return self.model

    def _extract_arrival_times(self):
        """
        Extract the arrival times at each task for each caregiver.
        """
        self.arrivals = {}
        for k in self.K:
            self.arrivals[k] = {}
            routes = self.routes[k]
            if not routes:
                continue
            first_task = routes[0][1]
            last_task = routes[-1][0]
            self.arrivals[k]["start"] = self.e[first_task] - self.c(k, "start", first_task)
            self.arrivals[k]["end"] = self.l[last_task] + self.c(k, last_task, "end")

            for _, j in self.routes[k]:
                if j == "end":
                    continue
                self.arrivals[k][j] = self.e[j]
        return self.arrivals
