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
        self.B = {}
        self.overtime = {}

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
                self.B[k, i] = self.model.addVar(vtype=GRB.BINARY, name=f"B^{k}_{i}")

            # Start and end times
            self.T[k, "start"] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_start")
            self.T[k, "end"] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"T^{k}_end")

            self.overtime[k] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"overtime_{k}")

            self.is_used[k] = self.model.addVar(vtype=GRB.BINARY, name=f"is_used_{k}")
        print("Created variables.")

        # ---- Objective Function
        caregiver_weight = 60
        self.model.setObjective(
            gp.quicksum(self.T[k, "end"] - self.T[k, "start"] for k in self.K)
            + 1.5 * gp.quicksum(self.overtime[k] for k in self.K)
            + caregiver_weight * gp.quicksum(self.is_used[k] for k in self.K),
            GRB.MINIMIZE,
        )
        print("Created objective function.")

        # ---- Constraints

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

        # One break break every 5 hours of work
        for k in self.K:
            self.model.addConstr(
                gp.quicksum(self.B[k, i] for i in self.V) >= (self.T[k, "end"] - self.T[k, "start"]) / (5 * 60) - 1,
                name=f"BreakRequired[{k}]",
            )

        # Can only take a break if the caregiver visited a task
        for k in self.K:
            for i in self.V:
                self.model.addConstr(
                    self.B[k, i]
                    <= gp.quicksum(self.feasible_breaks[k, i, j] * self.x[k, i, j] for j in self.V if j != i),
                    name=f"BreakVisit[{k},{i}]",
                )

        # Shift length constraints
        for k in self.K:
            # Start time before end time
            self.model.addConstr(self.T[k, "end"] >= self.T[k, "start"], name=f"StartBeforeEnd[{k}]")

            # Overtime calculation constraint - overtime is hours worked beyond 8 hours
            regular_hours = 8 * 60  # 8 hours in minutes
            self.model.addConstr(
                self.overtime[k] >= self.T[k, "end"] - self.T[k, "start"] - regular_hours,
                name=f"OvertimeCalculation[{k}]",
            )

        # Start and end time definitions
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
