import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .base_model import BaseModel


class FixedModel(BaseModel):
    def build(self):
        self.model = gp.Model("HomeCare")
        self.x = {}
        self.T = {}
        for k in self.K:
            for i in self.V:
                # Add route to the start and end nodes
                self.x[k, "start", i] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_start_{i}")
                self.x[k, i, "end"] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_end")
                for j in self.V:
                    if i != j:
                        self.x[k, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_{j}")

                self.T[k, "start"] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_start")
                self.T[k, "end"] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_end")

        # ---- Objective Function
        # Minimize time between start and end nodes for all caregivers
        self.model.setObjective(gp.quicksum(self.T[k, "end"] - self.T[k, "start"] for k in self.K), GRB.MINIMIZE)

        # ---- Constraints

        # Each task is visited exactly once by exactly one caregiver
        for i in self.V:
            self.model.addConstr(
                gp.quicksum(self.x[k, j, i] for k in self.K for j in self.V + ["start"] if j != i) == 1,
                name=f"UniqueVisit[{i}]",
            )

        # Flow conservation for each caregiver k
        for k in self.K:
            for i in self.V:
                self.model.addConstr(
                    gp.quicksum(self.x[k, i, j] for j in self.V + ["end"] if i != j)
                    - gp.quicksum(self.x[k, j, i] for j in self.V + ["start"] if i != j)
                    == 0,
                    name=f"Flow[{k},{i}]",
                )

        # Route completion (start and end usage) for each caregiver
        for k in self.K:
            self.model.addConstr(gp.quicksum(self.x[k, "start", i] for i in self.V) <= 1, name=f"StartBalance[{k}]")

        # Only visit clients that the caregiver is qualified to visit
        self.model.addConstr(
            gp.quicksum(
                self.x[k, i, j]
                for k in self.K
                for j in self.V
                for i in self.V + ["start"]
                if j not in self.caregiver_tasks[k] and i != j
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

        # Start time before end time
        for k in self.K:
            self.model.addConstr(self.T[k, "end"] >= self.T[k, "start"], name=f"TemporalFeasibility[{k}]")

        # Start and end time "definitions"
        for k in self.K:
            self.model.addConstr(
                self.T[k, "start"]
                <= gp.quicksum(self.x[k, "start", i] * (self.e[i] - self.c[k, "start", i]) for i in self.V),
                name=f"StartTime[{k}]",
            )
            self.model.addConstr(
                self.T[k, "end"]
                >= gp.quicksum(self.x[k, i, "end"] * (self.l[i] + self.c[k, i, "end"]) for i in self.V),
                name=f"EndTime[{k}]",
            )

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
