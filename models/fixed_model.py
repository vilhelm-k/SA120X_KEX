import gurobipy as gp
from gurobipy import GRB
from .base_model import BaseModel


class FixedModel(BaseModel):
    def build(
        self,
        overtime_penalty=1.5,
        caregiver_penalty=60,
        worktime_per_break=5 * 60,
        regular_hours=8 * 60,
        break_length=30,
        continuity_penalty=30,
    ):
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

                # Routes between tasks
                for j in self.V:
                    if i != j:
                        self.x[k, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_{j}")

            # Start and end times
            self.S[k] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_start")
            self.E[k] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_end")

            # Caregiver usage
            self.is_used[k] = self.model.addVar(vtype=GRB.BINARY, name=f"is_used_{k}")

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
        self.model.addConstr(
            gp.quicksum(
                self.x[k, i, j]
                for k in self.K
                for j in self.V
                for i in self.V + ["start"]
                if j not in self.Vk[k] and i != j
            )
            == 0,
            name="Qualification",
        )

        # Temporally infeasible tasks are not visited
        self.model.addConstr(
            gp.quicksum(
                self.x[k, i, j]
                for k in self.K
                for j in self.V
                for i in self.V
                if i != j and self.e[j] < self.l[i] + self.c[k, i, j]
            )
            == 0,
            name="TemporalInfeasibility",
        )

        # Start and end time definitions
        for k in self.K:
            self.model.addConstr(
                self.S[k] <= gp.quicksum(self.x[k, "start", i] * (self.e[i] - self.c[k, "start", i]) for i in self.V),
                name=f"StartTime[{k}]",
            )
            self.model.addConstr(
                self.E[k] >= gp.quicksum(self.x[k, i, "end"] * (self.l[i] + self.c[k, i, "end"]) for i in self.V),
                name=f"EndTime[{k}]",
            )

            # Start time before end time (basic temporal constraint)
            self.model.addConstr(self.E[k] >= self.S[k], name=f"StartBeforeEnd[{k}]")

        # Base objective: minimize total time
        base_objective = gp.quicksum(self.E[k] - self.S[k] for k in self.K)
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
            print("Adding break requirements.")
            self.B = {}
            feasible_breaks = self.determine_feasible_breaks(break_length)

            # Create break variables
            for k in self.K:
                for i in self.V:
                    self.B[k, i] = self.model.addVar(vtype=GRB.BINARY, name=f"B^{k}_{i}")

            # Add break constraints
            for k in self.K:
                # One break for every worktime_per_break minutes of work
                self.model.addConstr(
                    gp.quicksum(self.B[k, i] for i in self.V) >= (self.E[k] - self.S[k]) / worktime_per_break - 1,
                    name=f"BreakRequired[{k}]",
                )

                # Can only take a break if the caregiver visited a task
                for i in self.V:
                    self.model.addConstr(
                        self.B[k, i]
                        <= gp.quicksum(
                            feasible_breaks[k, i, j] * self.x[k, i, j] for j in self.V if (k, i, j) in feasible_breaks
                        ),
                        name=f"BreakVisit[{k},{i}]",
                    )

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
                    # If there's a historical assignment, new_serve must be 0
                    if self.H[k, c] == 1:
                        self.model.addConstr(self.new_serve[k, c] == 0, name=f"NoNewServe[{k},{c}]")
                        continue

                    for i in self.Vc[c]:
                        self.model.addConstr(
                            self.new_serve[k, c] >= gp.quicksum(self.x[k, i, j] for i in self.V + ["start"] if i != j),
                            name=f"NewServe[{k},{c}]",
                        )

            # Add continuity penalty term to objective
            objective_terms.append(
                continuity_penalty * gp.quicksum(self.new_serve[k, c] for k in self.K for c in self.C)
            )

        # Update objective function with all terms
        if len(objective_terms) > 1:  # If we added more terms beyond the base
            self.model.setObjective(sum(objective_terms), GRB.MINIMIZE)
            print("Updated objective function with penalties.")

        return self.model

    def _extract_routes(self):
        """
        Extracts the ordered route into a dictionary for each caregiver.
        """
        # First pass: build adjacency lists for each caregiver
        adjacency = {k: {} for k in self.K}

        for k, i, j in self.x:
            if self.x[k, i, j].X > 0.5:
                adjacency[k][i] = j

        # Second pass: traverse adjacency lists to build ordered routes
        routes = {k: [] for k in self.K}

        for k in self.K:
            current = "start"
            while current in adjacency[k] and current != "end":
                next_node = adjacency[k][current]
                routes[k].append((current, next_node))
                current = next_node

        self.routes = routes
        return routes
