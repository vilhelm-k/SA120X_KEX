import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .base_model import BaseModel


class BreakFixedModel(BaseModel):
    def build(self):
        self.model = gp.Model("HomeCare")
        self.x = {}
        self.T = {}
        self.is_used = {}
        self.y = {}  # Auxiliary variables for break sequences

        # Create decision variables
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

                # Routes to/from break (only from regular tasks)
                self.x[k, i, "break"] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_break")
                self.x[k, "break", i] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_break_{i}")

                # Auxiliary variables for break sequences (i→break→j)
                for j in self.V:
                    if i != j:
                        self.y[k, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"y^{k}_{i}_{j}")

            # Start and end times
            self.T[k, "start"] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_start")
            self.T[k, "end"] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_end")

            # Caregiver usage
            self.is_used[k] = self.model.addVar(vtype=GRB.BINARY, name=f"is_used_{k}")

        # ---- Objective Function
        caregiver_weight = 60
        self.model.setObjective(
            gp.quicksum(self.T[k, "end"] - self.T[k, "start"] for k in self.K)
            + caregiver_weight * gp.quicksum(self.is_used[k] for k in self.K),
            GRB.MINIMIZE,
        )

        # ---- Constraints

        # Define caregiver usage
        for k in self.K:
            self.model.addConstr(
                self.is_used[k] == gp.quicksum(self.x[k, "start", i] for i in self.V), name=f"Used[{k}]"
            )

        # Each task is visited exactly once by exactly one caregiver
        for i in self.V:
            self.model.addConstr(
                gp.quicksum(self.x[k, j, i] for k in self.K for j in self.V + ["start", "break"] if j != i) == 1,
                name=f"UniqueVisit[{i}]",
            )

        # Flow conservation for regular tasks
        for k in self.K:
            for i in self.V:
                self.model.addConstr(
                    gp.quicksum(self.x[k, i, j] for j in self.V + ["end", "break"] if j != i)
                    - gp.quicksum(self.x[k, j, i] for j in self.V + ["start", "break"] if j != i)
                    == 0,
                    name=f"Flow[{k},{i}]",
                )

        # Flow conservation for break nodes
        for k in self.K:
            self.model.addConstr(
                gp.quicksum(self.x[k, i, "break"] for i in self.V) - gp.quicksum(self.x[k, "break", j] for j in self.V)
                == 0,
                name=f"BreakFlow[{k}]",
            )

        # Only visit clients that the caregiver is qualified to visit
        self.model.addConstr(
            gp.quicksum(
                self.x[k, i, j]
                for k in self.K
                for j in self.V
                for i in self.V + ["start", "break"]
                if j not in self.caregiver_tasks[k] and i != j
            )
            == 0,
            name="Qualification",
        )

        # Link break sequence variables to x variables
        for k in self.K:
            for i in self.V:
                for j in self.V:
                    if i != j:
                        # y[k,i,j] = 1 iff x[k,i,"break"] = 1 and x[k,"break",j] = 1
                        self.model.addConstr(self.y[k, i, j] <= self.x[k, i, "break"], name=f"BreakSeq1[{k},{i},{j}]")
                        self.model.addConstr(self.y[k, i, j] <= self.x[k, "break", j], name=f"BreakSeq2[{k},{i},{j}]")
                        self.model.addConstr(
                            self.y[k, i, j] >= self.x[k, i, "break"] + self.x[k, "break", j] - 1,
                            name=f"BreakSeq3[{k},{i},{j}]",
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

        # Temporally infeasible break sequences are not allowed (using indicator constraints)
        for k in self.K:
            for i in self.V:
                for j in self.V:
                    if i != j:
                        # Check if sequence i→break→j is temporally infeasible
                        break_time = self.l[i] + self.s[i] + self.c[k, i, "break"] + 30 + self.c[k, "break", j]
                        if self.e[j] < break_time:
                            self.model.addGenConstrIndicator(
                                self.y[k, i, j],
                                1,  # When y[k,i,j] = 1
                                False,  # Constraint must be false (infeasible)
                                name=f"BreakSeqInfeasible[{k},{i},{j}]",
                            )

        # Required breaks based on shift length
        for k in self.K:
            self.model.addConstr(
                gp.quicksum(self.x[k, i, "break"] for i in self.V) >= (self.T[k, "end"] - self.T[k, "start"]) / 5 * 60,
                name=f"BreakRequirement[{k}]",
            )

        # Shift length constraints
        for k in self.K:
            # Start time before end time
            self.model.addConstr(self.T[k, "end"] >= self.T[k, "start"], name=f"StartBeforeEnd[{k}]")

            # Maximum shift length (12 hours)
            self.model.addConstr(self.T[k, "end"] - self.T[k, "start"] <= 12 * 60, name=f"MaxShiftLength[{k}]")

        # Start and end time definitions
        for k in self.K:
            self.model.addConstr(
                self.T[k, "start"]
                <= gp.quicksum(self.x[k, "start", i] * (self.e[i] - self.c[k, "start", i]) for i in self.V),
                name=f"StartTime[{k}]",
            )
            self.model.addConstr(
                self.T[k, "end"]
                >= gp.quicksum(self.x[k, i, "end"] * (self.l[i] + self.s[i] + self.c[k, i, "end"]) for i in self.V),
                name=f"EndTime[{k}]",
            )

        print("Model built with break constraints.")
        return self.model

    def _extract_arrival_times(self):
        """
        Extract the arrival times at each task for each caregiver.
        """
        self.arrivals = {}
        for k in self.K:
            self.arrivals[k] = {}
            self.arrivals[k]["start"] = self.T[k, "start"].X
            self.arrivals[k]["end"] = self.T[k, "end"].X

            for _, j in self.routes[k]:
                if j == "end":
                    continue
                self.arrivals[k][j] = self.e[j]
        return self.arrivals
