import gurobipy as gp
from gurobipy import GRB
from .base_model import BaseModel


class TSPModelStrict(BaseModel):
    def build(
        self,
        overtime_penalty=0.8,
        caregiver_penalty=60,
        worktime_per_break=5 * 60,
        regular_hours=8 * 60,
        break_length=30,
        continuity_penalty=30,
        day_continuity_penalty=8,
        lateness_penalty=5,
        break_penalty=100,
    ):
        print("Building TSP model.")
        # ---- Base Model Construction ----
        self.model = gp.Model("HomeCare")

        # Base variables
        self.x = {}  # Route variables
        self.w = {}  # Weights incorporating service times and waiting times, measured end to end.
        self.T = {}  # Total time variables
        self.is_used = {}  # Caregiver usage variables

        # Create base decision variables
        for k in self.K:
            # Regular task routes
            for i in self.V:
                # Routes between tasks and start/end
                self.x[k, "start", i] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_start_{i}")
                self.w[k, "start", i] = self.c(k, "start", i) + self.s[i]
                self.x[k, i, "end"] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_end")
                self.w[k, i, "end"] = self.c(k, i, "end")

                # Routes between tasks
                for j in self.V:
                    if i != j:
                        self.x[k, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_{j}")
                        self.w[k, i, j] = self.l[j] - self.l[i]

            # Caregiver usage
            self.is_used[k] = self.model.addVar(vtype=GRB.BINARY, name=f"is_used_{k}")

            # Total time
            self.T[k] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"T_{k}")

        print("Created base variables.")

        # ---- Base Constraints ----

        # Define caregiver usage
        for k in self.K:
            self.model.addConstr(
                self.is_used[k] == gp.quicksum(self.x[k, "start", i] for i in self.V), name=f"Used[{k}]"
            )

        # Each task is visited exactly once by exactly one caregiver
        for i in self.V:
            self.model.addConstr(
                gp.quicksum(self.x[k, j, i] for k in self.K for j in self.V + ["start"] if j != i) == 1,
                name=f"UniqueVisit[{i}]",
            )

        # Flow conservation for regular tasks
        for k in self.K:
            for i in self.V:
                self.model.addConstr(
                    gp.quicksum(self.x[k, i, j] for j in self.V + ["end"] if j != i)
                    - gp.quicksum(self.x[k, j, i] for j in self.V + ["start"] if j != i)
                    == 0,
                    name=f"Flow[{k},{i}]",
                )

        # Only visit clients that the caregiver is qualified to visit
        unqualified_visits = []
        for k in self.K:
            for j in self.V:
                if not self.is_caregiver_qualified(k, j):
                    for i in self.V + ["start"]:
                        if i != j:
                            unqualified_visits.append(self.x[k, i, j])

        if unqualified_visits:
            self.model.addConstr(
                gp.quicksum(unqualified_visits) == 0,
                name="Qualification",
            )

        # Temporal feasibility constraint
        # If arrival time window is infeasible (e[j] - l[i] - travel_time < 0), force x[k,i,j] to be 0
        infeasible_arcs = []
        for k in self.K:
            for i in self.V:
                for j in self.V:
                    if i != j:
                        # Check if it's temporally infeasible to go from i to j
                        if self.e[j] - self.l[i] - self.c(k, i, j) < 0:
                            infeasible_arcs.append(self.x[k, i, j])

        if infeasible_arcs:
            self.model.addConstr(gp.quicksum(infeasible_arcs) == 0, name="TemporalFeasibility")

        # Define total time for each caregiver
        for k in self.K:
            self.model.addConstr(
                self.T[k]
                >= gp.quicksum(self.w[k, i, j] * self.x[k, i, j] for i in self.V for j in self.V if i != j)
                + gp.quicksum(
                    self.w[k, "start", i] * self.x[k, "start", i] + self.w[k, i, "end"] * self.x[k, i, "end"]
                    for i in self.V
                ),
                name=f"TotalTime[{k}]",
            )

        # Base objective: minimize total time
        base_objective = gp.quicksum(self.T[k] for k in self.K)
        self.model.setObjective(base_objective, GRB.MINIMIZE)

        print("Built base model.")

        # ---- Optional Components ----
        objective_terms = [base_objective]

        # 1. Add overtime penalties if needed
        if overtime_penalty > 0:
            print("Adding overtime penalties.")
            self.overtime = {}

            # Create overtime variables
            for k in self.K:
                self.overtime[k] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"overtime_{k}")

                # Overtime calculation constraint
                self.model.addConstr(
                    self.overtime[k] >= self.T[k] - regular_hours,
                    name=f"OvertimeCalculation[{k}]",
                )

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
            self.B = {}
            self.missed_breaks = {}

            # Create evening shift variables (1 if caregiver starts after 15:30 = 930 minutes)
            self.is_evening_shift = {}
            evening_shift_time = 15 * 60 + 30  # 15:30 in minutes since midnight

            for k in self.K:
                self.is_evening_shift[k] = self.model.addVar(vtype=GRB.BINARY, name=f"is_evening_shift_{k}")
                self.missed_breaks[k] = self.model.addVar(vtype=GRB.INTEGER, lb=0, name=f"missed_breaks_{k}")

                # Set is_evening_shift to 1 if caregiver starts after 15:30
                start_time = gp.quicksum(self.x[k, "start", i] * (self.e[i] - self.c(k, "start", i)) for i in self.V)
                self.model.addConstr(
                    self.is_evening_shift[k] <= (start_time - evening_shift_time) / 1440 + 1,
                    name=f"EveningShiftUpper[{k}]",
                )

            # Create break variables
            for k in self.K:
                for i in self.V:
                    self.B[k, i] = self.model.addVar(vtype=GRB.BINARY, name=f"B^{k}_{i}")

            # Add break constraints with evening shift exemption
            for k in self.K:
                # One break for every worktime_per_break minutes of work, but only if not on evening shift
                self.model.addGenConstrIndicator(
                    self.is_evening_shift[k],
                    False,
                    self.missed_breaks[k]
                    >= (self.T[k] / worktime_per_break - 1) - gp.quicksum(self.B[k, i] for i in self.V),
                    name=f"BreakRequired[{k}]",
                )

                # Can only take a break if the caregiver visited a task with enough time for the break
                for i in self.V:
                    # Collect all valid break pairs
                    valid_break_terms = []
                    for j in self.V:
                        if i != j and self.is_break_feasible(k, i, j, break_length):
                            valid_break_terms.append(self.x[k, i, j])

                    if valid_break_terms:
                        self.model.addConstr(
                            self.B[k, i] <= gp.quicksum(valid_break_terms),
                            name=f"BreakVisit[{k},{i}]",
                        )
                    else:
                        # If no break is possible after this task, set the break variable to 0
                        self.model.addConstr(self.B[k, i] == 0, name=f"NoBreak[{k},{i}]")

            # Add missed breaks penalty to objective
            objective_terms.append(break_penalty * gp.quicksum(self.missed_breaks[k] for k in self.K))

        # 4. Add continuity of care penalties if needed
        if continuity_penalty > 0:
            print("Adding continuity of care penalties.")
            self.new_serve = {}

            # Create new_serve variables
            for k in self.K:
                for c in self.C:
                    self.new_serve[k, c] = self.model.addVar(vtype=GRB.BINARY, name=f"new_serve_{k}_{c}")

            # Add constraints to detect new assignments
            for k in self.K:
                for c in self.C:
                    client_tasks = self.get_client_tasks(c)
                    for j in client_tasks:
                        self.model.addConstr(
                            self.new_serve[k, c] >= gp.quicksum(self.x[k, i, j] for i in self.V + ["start"] if i != j),
                            name=f"NewServe[{k},{c}]",
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
