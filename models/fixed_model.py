import gurobipy as gp
from gurobipy import GRB
from .base_model import BaseModel


class FixedModel(BaseModel):
    def build(
        self,
        overtime_penalty=0.5,
        caregiver_penalty=60,
        worktime_per_break=5 * 60,
        regular_hours=8 * 60,
        break_length=30,
        continuity_penalty=35,
        day_continuity_penalty=10,
        lateness_penalty=5,
        warm_start=False,
        break_penalty=100,
    ):
        print("Building fixed model.")
        # ---- Base Model Construction ----
        self.model = gp.Model("HomeCare")

        # Base variables
        self.x = {}  # Route variables
        self.E = {}  # End time variables
        self.S = {}  # Start time variables
        self.is_used = {}  # Caregiver usage variables

        # Create base decision variables
        for k in self.K:
            # Regular task routes
            for i in self.V:
                # Routes between tasks and start/end
                self.x[k, "start", i] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_start_{i}")
                self.x[k, i, "end"] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_end")

                # Set warm start values for start/end routes if enabled
                if warm_start:
                    # Check if this edge exists in the real route
                    exists_start_i = any(edge == ("start", i) for edge in self.real_routes[k])
                    exists_i_end = any(edge == (i, "end") for edge in self.real_routes[k])

                    if exists_start_i:
                        self.x[k, "start", i].Start = 1
                    else:
                        self.x[k, "start", i].Start = 0

                    if exists_i_end:
                        self.x[k, i, "end"].Start = 1
                    else:
                        self.x[k, i, "end"].Start = 0

                # Routes between tasks
                for j in self.V:
                    if i != j:
                        self.x[k, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_{j}")

                        # Set warm start values for task-to-task routes
                        if warm_start:
                            # Check if this edge exists in the real route
                            exists_ij = any(edge == (i, j) for edge in self.real_routes[k])
                            self.x[k, i, j].Start = 1 if exists_ij else 0

            # Start and end times
            self.S[k] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_start")
            self.E[k] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_end")

            # Set warm start values for start and end times
            if warm_start and k in self.real_arrivals and "start" in self.real_arrivals[k]:
                self.S[k].Start = self.real_arrivals[k]["start"]
                self.E[k].Start = self.real_arrivals[k]["end"]

            # Caregiver usage
            self.is_used[k] = self.model.addVar(vtype=GRB.BINARY, name=f"is_used_{k}")

            # Set warm start value for caregiver usage
            if warm_start:
                self.is_used[k].Start = 1 if self.real_routes[k] else 0

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

        # Temporal feasibility constraint without slack variables when not needed
        # Instead, we'll directly add the constant term to the objective when needed
        # Calculate lateness penalty directly in the objective function
        # This is more efficient than accumulating terms one by one
        lateness_total = gp.quicksum(
            -1 * (self.e[j] - self.l[i] - self.c(k, i, j)) * self.x[k, i, j]
            for k in self.K
            for i in self.V
            for j in self.V
            if i != j and (self.e[j] - self.l[i] - self.c(k, i, j)) < 0
        )

        # Start and end time definitions
        for k in self.K:
            self.model.addConstr(
                self.S[k] <= gp.quicksum(self.x[k, "start", i] * (self.e[i] - self.c(k, "start", i)) for i in self.V),
                name=f"StartTime[{k}]",
            )
            self.model.addConstr(
                self.E[k] >= gp.quicksum(self.x[k, i, "end"] * (self.l[i] + self.c(k, i, "end")) for i in self.V),
                name=f"EndTime[{k}]",
            )

            # Start time before end time (basic temporal constraint)
            self.model.addConstr(self.E[k] >= self.S[k], name=f"StartBeforeEnd[{k}]")

        # Base objective: minimize total time + lateness penalty
        base_objective = gp.quicksum(self.E[k] - self.S[k] for k in self.K) + lateness_penalty * lateness_total
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

                # Set warm start values for overtime
                if warm_start and k in self.real_arrivals and "start" in self.real_arrivals[k]:
                    overtime_value = max(
                        0, self.real_arrivals[k]["end"] - self.real_arrivals[k]["start"] - regular_hours
                    )
                    self.overtime[k].Start = overtime_value

                # Overtime calculation constraint
                self.model.addConstr(
                    self.overtime[k] >= self.E[k] - self.S[k] - regular_hours,
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

                # Set is_evening_shift to 1 if caregiver starts after 15:30
                self.model.addConstr(
                    self.is_evening_shift[k] <= (self.S[k] - evening_shift_time) / 1440 + 1,
                    name=f"EveningShiftUpper[{k}]",
                )

                self.missed_breaks[k] = self.model.addVar(vtype=GRB.INTEGER, lb=0, name=f"missed_breaks_{k}")

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
                    >= ((self.E[k] - self.S[k]) / worktime_per_break - 1) - gp.quicksum(self.B[k, i] for i in self.V),
                    name=f"BreakRequired[{k}]",
                )

                # Can only take a break if the caregiver visited a task
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
