import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .base_model import BaseModel


class BasicFlexibleModel(BaseModel):
    def __calculate_big_M(self):
        """
        Calculates the big M for the models.
        """
        earliest_start = self.tasks["start_minutes"].min()
        latest_end = self.tasks["end_minutes"].max()
        max_travel_time = max(self.c.values())
        M = 1.1 * (latest_end - earliest_start + 2 * max_travel_time)
        return M

    def build(self):
        big_M = self.__calculate_big_M()
        self.model = gp.Model("HomeCare")
        self.x = {}
        for k in self.K:
            for i in self.V:
                # Add route to the start and end nodes
                self.x[k, "start", i] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_start_{i}")
                self.x[k, i, "end"] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_end")
                for j in self.V:
                    if i != j:
                        self.x[k, i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_{j}")

        # t[k,i] = arrival time of caregiver k at node i
        self.t = {}
        for k in self.K:
            self.t[k, "start"] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"t^{k}_start")
            self.t[k, "end"] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"t^{k}_end")
            for i in self.V:
                self.t[k, i] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"t^{k}_{i}")

        self.model.update()

        # ---- Objective Function
        # Minimize time between start and end nodes for all caregivers
        self.model.setObjective(gp.quicksum(self.t[k, "end"] - self.t[k, "start"] for k in self.K), GRB.MINIMIZE)

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

        # Arriving within time windows
        for k in self.K:
            for i in self.V:
                self.model.addConstr(self.t[k, i] >= self.e[i], name=f"Earliest[{k},{i}]")
                self.model.addConstr(self.t[k, i] <= self.l[i] - self.s[i], name=f"Latest[{k},{i}]")

        # Temporal feasibility
        for k in self.K:
            self.model.addConstr(self.t[k, "end"] >= self.t[k, "start"], name=f"TemporalFeasibility[{k}]")
            for i in self.V + ["start"]:
                for j in self.V + ["end"]:
                    if i != j and not (i == "start" and j == "end"):
                        service_time = 0 if i == "start" else self.s[i]
                        travel_time = self.c[k, i, j]
                        self.model.addConstr(
                            self.t[k, j] >= self.t[k, i] + travel_time + service_time - big_M * (1 - self.x[k, i, j]),
                            name=f"TimeLink[{k},{i}->{j}]",
                        )

        return self.model

    def _extract_arrival_times(self):
        """
        Extract the arrival times at each task for each caregiver.
        """
        arrivals = {}
        for k in self.K:
            arrivals[k] = {}
            arrivals[k]["start"] = self.t[k, "start"].X

            # Extract arrivals for tasks in the route
            for _, j in self.routes[k]:
                arrivals[k][j] = self.t[k, j].X

        self.arrivals = arrivals
        return arrivals
