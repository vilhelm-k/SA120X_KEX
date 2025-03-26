import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .base_model import BaseModel


class BasicFixedModel(BaseModel):
    def build(self):
        self.model = gp.Model("HomeCare")
        self.x = {}
        self.T = {}
        self.is_used = {}
        for k in self.K:
            for i in self.V:
                # Add route to the start and end nodes
                self.x[k, "start", i] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_start_{i}")
                self.x[k, i, "end"] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_end")
                for j in self.V:
                    if i != j:
                        self.x[k, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_{j}")

            # Start and end times for each caregiver
            self.T[k, "start"] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_start")
            self.T[k, "end"] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_end")

            # Define if caregiver k is used
            self.is_used[k] = self.model.addVar(vtype=GRB.BINARY, name=f"is_used_{k}")
            self.model.addConstr(
                self.is_used[k] == gp.quicksum(self.x[k, "start", i] for i in self.V), name=f"Used[{k}]"
            )

        # ---- Objective Function
        # Minimize time between start and end nodes for all caregivers
        caregiver_weight = 60
        self.model.setObjective(
            gp.quicksum(self.T[k, "end"] - self.T[k, "start"] for k in self.K)
            + caregiver_weight * gp.quicksum(self.is_used[k] for k in self.K),
            GRB.MINIMIZE,
        )

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

            self.model.addConstr(self.T[k, "end"] - self.T[k, "start"] <= 12 * 60, name=f"TemporalFeasibility[{k}]")

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
        print("Model built.")
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
