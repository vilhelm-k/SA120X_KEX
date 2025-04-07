import gurobipy as gp
from gurobipy import GRB
from .base_model import BaseModel


class PartitionModel(BaseModel):
    def build(
        self,
        caregiver_penalty=60,
        overtime_penalty=1.5,
        worktime_per_break=5 * 60,
        continuity_penalty=30,
        regular_hours=8 * 60,
        break_length=30,
    ):
        print("Building partition model...")
        # ---- Base Model Construction ----
        self.model = gp.Model("PartitionHomeCareModel")

        # Base variables
        self.x = {}  # Route variables
        self.S = {}  # Start time variables
        self.E = {}  # End time variables
        self.is_used = {}  # Caregiver usage variables

        # Create base decision variables
        for k in self.K:
            for i in self.V:
                # Caregiver k visits task i
                self.x[k, i] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}")

            # Start and end times
            self.S[k] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"S^{k}")
            self.E[k] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"E^{k}")

            # Caregiver usage
            self.is_used[k] = self.model.addVar(vtype=GRB.BINARY, name=f"is_used_{k}")

        print("Created base variables.")

        # ---- Base Constraints ----

        # Define caregiver usage
        for k in self.K:
            self.model.addGenConstrOr(self.is_used[k], [self.x[k, i] for i in self.V], name=f"Used[{k}]")

        # Each task is visited exactly once by exactly one caregiver
        for i in self.V:
            self.model.addConstr(gp.quicksum(self.x[k, i] for k in self.K) == 1, name=f"UniqueVisit[{i}]")

        # Only visit clients that the caregiver is qualified to visit
        self.model.addConstr(
            gp.quicksum(self.x[k, i] for k in self.K for i in self.V if i not in self.Vk[k]) == 0,
            name="Qualification",
        )

        # Temporally infeasible pairs of tasks are not possible
        for k, i, j in self.A:
            if self.A[k, i, j] == 0:
                self.model.addConstr(self.x[k, i] + self.x[k, j] <= 1, name=f"InfeasiblePair[{k},{i},{j}]")

        # Start and end time definitions
        for k in self.K:
            self.model.addConstr(self.E[k] >= self.S[k], name=f"StartBeforeEnd[{k}]")
            for i in self.V:
                self.model.addGenConstrIndicator(
                    self.x[k, i],
                    True,
                    self.S[k] <= self.e[i] - self.c(k, "start", i),
                    name=f"StartTime[{k},{i}]",
                )
                self.model.addGenConstrIndicator(
                    self.x[k, i],
                    True,
                    self.E[k] >= self.l[i] + self.c(k, i, "end"),
                    name=f"EndTime[{k},{i}]",
                )
        print("Built base model.")

        # Base objective: minimize total time
        objective_terms = [gp.quicksum(self.E[k] - self.S[k] for k in self.K)]

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
            for k, i, j in feasible_breaks:
                if not feasible_breaks[k, i, j]:
                    self.model.addConstr(self.B[k, i] + self.x[k, j] <= 1, name=f"BreakVisit[{k},{i},{j}]")

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
                    for i in self.Vc[c]:
                        self.model.addConstr(
                            self.new_serve[k, c] >= self.x[k, i] - self.H[k, c],
                            name=f"NewServe[{k},{c},{i}]",
                        )

            # Add continuity penalty term to objective
            objective_terms.append(
                continuity_penalty * gp.quicksum(self.new_serve[k, c] for k in self.K for c in self.C)
            )

        self.model.setObjective(sum(objective_terms), GRB.MINIMIZE)
        return self.model

    def _extract_routes(self):
        """
        Extract the routes for each caregiver from the model.
        """
        self.routes = {}
        for k in self.K:
            tasks = []
            for i in self.V:
                if self.x[k, i].X > 0.5:
                    tasks.append(i)
            if not tasks:
                continue
            tasks.sort(key=lambda i: self.e[i])

            self.routes[k] = [("start", tasks[0])]
            for i in range(len(tasks) - 1):
                self.routes[k].append((tasks[i], tasks[i + 1]))
            self.routes[k].append((tasks[-1], "end"))
        return self.routes
