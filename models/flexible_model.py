import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def calculate_big_M(tasks, time_matrices):
    """
    Calculates the big M for the models.
    """
    earliest_start = tasks["start_minutes"].min()
    latest_end = tasks["end_minutes"].max()
    max_travel_time = max([time_matrix.max().max() for time_matrix in time_matrices])
    M = 1.1 * (latest_end - earliest_start + 2 * max_travel_time)
    return M


def build_model(
    caregivers: pd.DataFrame,  # ID,ModeOfTransport,Attributes,start_minutes,end_minutes,StartLocation,EndLocation,RequiresBreak
    tasks: pd.DataFrame,  # ID,ClientID,start_minutes,end_minutes,duration_minutes,TaskType,PlannedCaregiverID
    clients: pd.DataFrame,  # ID,Requirements
    drive_time_matrix: pd.DataFrame,  # Drive time matrix from client col to client row. Uses client ID as indexes. Index 0 is the HQ.
    walk_time_matrix: pd.DataFrame,
    bicycle_time_matrix: pd.DataFrame,
):
    K = caregivers.index.tolist()
    V = tasks.index.tolist()
    M = calculate_big_M(tasks, [drive_time_matrix, walk_time_matrix, bicycle_time_matrix])
    model = gp.Model("HomeCare")

    # For each caregiver, gather only the patients that caregiver k can serve,
    # then define the augmented node set (start, qualified patients, end).
    caregiver_tasks = {}
    for k in K:
        caregiver_attributes = caregivers.loc[k, "Attributes"]
        qualified_patients = clients[
            clients["Requirements"].apply(lambda req: np.dot(req, caregiver_attributes) == 0)
        ].index.tolist()
        # Filter tasks to only include the ones with these qualified patients
        caregiver_tasks[k] = tasks[tasks["ClientID"].isin(qualified_patients)].index.tolist()

    # ---- 2. Decision Variables

    # x[k, i, j] = 1 if caregiver k goes directly from i to j, else 0.
    # Skip arcs into sigma[k] or out of tau[k], and skip i->i.
    x = {}
    for k in K:
        for i in V:
            # Add route to the start and end nodes
            x[k, "start", i] = model.addVar(vtype=GRB.BINARY, name=f"x^{k}_start_{i}")
            x[k, i, "end"] = model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_end")
            for j in V:
                if i != j:
                    x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x^{k}_{i}_{j}")

    # t[k,i] = arrival time of caregiver k at node i
    t = {}
    for k in K:
        t[k, "start"] = model.addVar(vtype=GRB.CONTINUOUS, name=f"t^{k}_start")
        t[k, "end"] = model.addVar(vtype=GRB.CONTINUOUS, name=f"t^{k}_end")
        for i in V:
            t[k, i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"t^{k}_{i}")

    model.update()

    # ---- 3. Objective Function
    # Minimize time bewteen start and end nodes for all caregivers
    model.setObjective(gp.quicksum(t[k, "end"] - t[k, "start"] for k in K), GRB.MINIMIZE)

    # ---- 4. Constraints
    # (V2) Each task is visited exactly once by exactly one caregiver
    for i in V:
        model.addConstr(
            gp.quicksum(x[k, j, i] for k in K for j in V + ["start"] if j != i) == 1, name=f"UniqueVisit[{i}]"
        )

    # (V3) Flow conservation for each caregiver k
    for k in K:
        for i in V:
            model.addConstr(
                gp.quicksum(x[k, i, j] for j in V + ["end"] if i != j)
                - gp.quicksum(x[k, j, i] for j in V + ["start"] if i != j)
                == 0,
                name=f"Flow[{k},{i}]",
            )

    # (V4) Route completion (start and end usage) for each caregiver
    # Only need to fix this for the start node, since flow conservation
    # and one visit per task ensures that the end node is also correctly handled
    for k in K:
        model.addConstr(gp.quicksum(x[k, "start", i] for i in V) <= 1, name=f"StartBalance[{k}]")

    # (V6) Only visit patients that the caregiver is qualified to visit
    model.addConstr(
        gp.quicksum(x[k, i, j] for k in K for j in V for i in V + ["start"] if j not in caregiver_tasks[k] and i != j)
        == 0,
        name="Qualification",
    )

    # (V7-V8) Arriving on time
    for k in K:
        for i in V:
            start_minutes = tasks.loc[i, "start_minutes"]
            end_minutes = tasks.loc[i, "end_minutes"]
            duration_minutes = tasks.loc[i, "duration_minutes"]
            model.addConstr(t[k, i] >= start_minutes, name=f"Earliest[{k},{i}]")
            model.addConstr(t[k, i] <= end_minutes - duration_minutes, name=f"Latest[{k},{i}]")

    # (V9) Temporal feasibility
    for k in K:
        model.addConstr(t[k, "end"] >= t[k, "start"], name=f"TemporalFeasibility[{k}]")
        for i in V + ["start"]:
            for j in V + ["end"]:
                if i != j and not (i == "start" and j == "end"):
                    # Calculate travel time based on locations
                    if i == "start":
                        # From start location to task
                        travel_time = (
                            0
                            if caregivers.loc[k, "StartLocation"] == "Home"
                            else drive_time_matrix.loc[0, tasks.loc[j, "ClientID"]]
                        )
                        service_time = 0
                    elif j == "end":
                        # From task to end location
                        travel_time = (
                            0
                            if caregivers.loc[k, "EndLocation"] == "Home"
                            else drive_time_matrix.loc[tasks.loc[i, "ClientID"], 0]
                        )
                        service_time = tasks.loc[i, "duration_minutes"]
                    else:
                        # From task to task
                        travel_time = drive_time_matrix.loc[tasks.loc[i, "ClientID"], tasks.loc[j, "ClientID"]]
                        service_time = tasks.loc[i, "duration_minutes"]

                    # Add the constraint
                    model.addConstr(
                        t[k, j] >= t[k, i] + travel_time + service_time - M * (1 - x[k, i, j]),
                        name=f"TimeLink[{k},{i}->{j}]",
                    )
    return model, x, t
