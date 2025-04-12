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
        print("Building fixed model...")
        self.model = gp.Model("HomeCare")

        # ----- VARIABLE CREATION -----
        print("Creating variable indices...")
        # Precompute all variable indices and warm start values
        x_indices = []
        x_start_vals = {}

        # Create indices for route variables
        for k in self.K:
            for i in self.V + ["start"]:
                for j in self.V + ["end"]:
                    if i != j and (i != "end" and j != "start"):
                        x_indices.append((k, i, j))

                        # Set warm start value if available
                        if warm_start and k in self.real_routes:
                            x_start_vals[(k, i, j)] = 1.0 if (i, j) in self.real_routes[k] else 0.0

        # Prepare start and end time values
        S_start_vals = {}
        E_start_vals = {}
        is_used_start_vals = {}

        for k in self.K:
            if warm_start and k in self.real_arrivals:
                if "start" in self.real_arrivals[k]:
                    S_start_vals[k] = self.real_arrivals[k]["start"]
                if "end" in self.real_arrivals[k]:
                    E_start_vals[k] = self.real_arrivals[k]["end"]
                is_used_start_vals[k] = 1.0 if self.real_routes.get(k, []) else 0.0

        print("Creating batch variables...")
        # Create all route variables in batch
        if warm_start:
            self.x = self.model.addVars(
                x_indices, vtype=GRB.BINARY, name="x", start={idx: x_start_vals.get(idx, 0.0) for idx in x_indices}
            )
        else:
            self.x = self.model.addVars(x_indices, vtype=GRB.BINARY, name="x")

        # Create time and usage variables in batch
        if warm_start:
            self.S = self.model.addVars(
                self.K, vtype=GRB.CONTINUOUS, name="T_start", start={k: S_start_vals.get(k, 0.0) for k in self.K}
            )
            self.E = self.model.addVars(
                self.K, vtype=GRB.CONTINUOUS, name="T_end", start={k: E_start_vals.get(k, 0.0) for k in self.K}
            )
            self.is_used = self.model.addVars(
                self.K, vtype=GRB.BINARY, name="is_used", start={k: is_used_start_vals.get(k, 0.0) for k in self.K}
            )
        else:
            self.S = self.model.addVars(self.K, vtype=GRB.CONTINUOUS, name="T_start")
            self.E = self.model.addVars(self.K, vtype=GRB.CONTINUOUS, name="T_end")
            self.is_used = self.model.addVars(self.K, vtype=GRB.BINARY, name="is_used")

        print("Created all variables in batch.")

        # ----- CONSTRAINT CREATION -----
        print("Adding constraints in batches...")

        # 1. Caregiver usage constraints
        usage_constrs = {k: self.is_used[k] == gp.quicksum(self.x[k, "start", i] for i in self.V) for k in self.K}
        self.model.addConstrs(usage_constrs, name="Used")

        # 2. Each task is visited exactly once
        visit_constrs = {
            i: gp.quicksum(self.x[k, j, i] for k in self.K for j in self.V + ["start"] if j != i) == 1 for i in self.V
        }
        self.model.addConstrs(visit_constrs, name="UniqueVisit")

        # 3. Flow conservation constraints
        flow_constrs = {
            (k, i): (
                gp.quicksum(self.x[k, i, j] for j in self.V + ["end"] if j != i)
                - gp.quicksum(self.x[k, j, i] for j in self.V + ["start"] if j != i)
            )
            == 0
            for k in self.K
            for i in self.V
        }
        self.model.addConstrs(flow_constrs, name="Flow")

        # 4. Only visit clients that the caregiver is qualified to visit
        # Collect all qualification constraints in a dictionary
        qualification_constrs = {}
        idx = 0
        for k in self.K:
            for j in self.V:
                if not self.is_caregiver_qualified(k, j):
                    for i in self.V + ["start"]:
                        if i != j and (k, i, j) in self.x:
                            qualification_constrs[idx] = self.x[k, i, j] == 0
                            idx += 1

        # Add all qualification constraints in batch if any exist
        if qualification_constrs:
            self.model.addConstrs(qualification_constrs, name="Qualification")

        # 5. Temporal feasibility and lateness penalties
        # Collect all lateness terms for the objective
        lateness_terms = []
        temporal_constrs = {}
        idx = 0

        for k in self.K:
            for i in self.V:
                for j in self.V:
                    if i != j and (k, i, j) in self.x:
                        time_diff = self.e[j] - self.l[i] - self.c(k, i, j)
                        if time_diff < 0:
                            lateness_terms.append(-1 * time_diff * self.x[k, i, j])

        lateness_total = gp.quicksum(lateness_terms)

        # 6. Start and end time constraints
        start_time_constrs = {
            k: self.S[k] <= gp.quicksum(self.x[k, "start", i] * (self.e[i] - self.c(k, "start", i)) for i in self.V)
            for k in self.K
        }
        end_time_constrs = {
            k: self.E[k] >= gp.quicksum(self.x[k, i, "end"] * (self.l[i] + self.c(k, i, "end")) for i in self.V)
            for k in self.K
        }
        start_end_constrs = {k: self.E[k] >= self.S[k] for k in self.K}

        self.model.addConstrs(start_time_constrs, name="StartTime")
        self.model.addConstrs(end_time_constrs, name="EndTime")
        self.model.addConstrs(start_end_constrs, name="StartBeforeEnd")

        # Base objective: minimize total time + lateness penalty
        base_objective = gp.quicksum(self.E[k] - self.S[k] for k in self.K) + lateness_penalty * lateness_total
        self.model.setObjective(base_objective, GRB.MINIMIZE)

        print("Built base model.")

        # ----- OPTIONAL COMPONENTS -----
        objective_terms = [base_objective]

        # 1. Add overtime penalties if needed
        if overtime_penalty > 0:
            print("Adding overtime penalties...")

            # Prepare overtime start values
            overtime_start_vals = {}
            if warm_start:
                for k in self.K:
                    if k in self.real_arrivals and "start" in self.real_arrivals[k] and "end" in self.real_arrivals[k]:
                        overtime_value = max(
                            0, self.real_arrivals[k]["end"] - self.real_arrivals[k]["start"] - regular_hours
                        )
                        overtime_start_vals[k] = overtime_value

            # Create overtime variables in batch
            if warm_start:
                self.overtime = self.model.addVars(
                    self.K,
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    name="overtime",
                    start={k: overtime_start_vals.get(k, 0.0) for k in self.K},
                )
            else:
                self.overtime = self.model.addVars(self.K, vtype=GRB.CONTINUOUS, lb=0, name="overtime")

            # Add overtime constraints in batch
            overtime_constrs = {k: self.overtime[k] >= self.E[k] - self.S[k] - regular_hours for k in self.K}
            self.model.addConstrs(overtime_constrs, name="OvertimeCalculation")

            # Add overtime term to objective
            objective_terms.append(overtime_penalty * gp.quicksum(self.overtime[k] for k in self.K))

        # 2. Add caregiver penalty if needed
        if caregiver_penalty > 0:
            print("Adding caregiver usage penalties...")
            # Add caregiver penalty term to objective
            objective_terms.append(caregiver_penalty * gp.quicksum(self.is_used[k] for k in self.K))

        # 3. Add break requirements if needed
        if worktime_per_break > 0:
            print("Adding break requirements with evening shift exemption...")

            # Create evening shift variables in batch
            self.is_evening_shift = self.model.addVars(self.K, vtype=GRB.BINARY, name="is_evening_shift")

            # Create missed breaks variables in batch
            self.missed_breaks = self.model.addVars(self.K, vtype=GRB.INTEGER, lb=0, name="missed_breaks")

            # Evening shift time threshold
            evening_shift_time = 15 * 60 + 30  # 15:30 in minutes since midnight

            # Set evening shift constraints in batch
            evening_constrs = {
                k: self.is_evening_shift[k] <= (self.S[k] - evening_shift_time) / 1440 + 1 for k in self.K
            }
            self.model.addConstrs(evening_constrs, name="EveningShiftUpper")

            # Create break variables in batch
            B_indices = [(k, i) for k in self.K for i in self.V]
            self.B = self.model.addVars(B_indices, vtype=GRB.BINARY, name="B")

            # Add break constraints with evening shift exemption
            # We need to add indicator constraints individually since they can't be batched
            for k in self.K:
                # One break for every worktime_per_break minutes, but only if not on evening shift
                self.model.addGenConstrIndicator(
                    self.is_evening_shift[k],
                    False,
                    self.missed_breaks[k]
                    >= ((self.E[k] - self.S[k]) / worktime_per_break - 1) - gp.quicksum(self.B[k, i] for i in self.V),
                    name=f"BreakRequired[{k}]",
                )

            # Can only take a break if caregiver visited a task with enough break time after
            break_constrs = {}
            for k in self.K:
                for i in self.V:
                    # Collect all valid break pairs
                    valid_break_terms = []
                    for j in self.V:
                        if i != j and self.is_break_feasible(k, i, j, break_length) and (k, i, j) in self.x:
                            valid_break_terms.append(self.x[k, i, j])

                    if valid_break_terms:
                        break_constrs[k, i] = self.B[k, i] <= gp.quicksum(valid_break_terms)
                    else:
                        break_constrs[k, i] = self.B[k, i] == 0

            # Add all break constraints in batch
            self.model.addConstrs(break_constrs, name="BreakVisit")

            # Add missed breaks penalty to objective
            objective_terms.append(break_penalty * gp.quicksum(self.missed_breaks[k] for k in self.K))

        # 4. Add continuity of care penalties if needed
        if continuity_penalty > 0:
            print("Adding continuity of care penalties...")

            # Create new_serve variables in batch
            new_serve_indices = [(k, c) for k in self.K for c in self.C]
            self.new_serve = self.model.addVars(new_serve_indices, vtype=GRB.BINARY, name="new_serve")

            # Add constraints to detect new assignments - organized for batch creation
            serve_constrs = {}

            for k in self.K:
                for c in self.C:
                    client_tasks = self.get_client_tasks(c)
                    for j in client_tasks:
                        if j in self.V:  # Ensure task is in current tasks
                            serve_key = (k, c, j)
                            serve_terms = [
                                self.x[k, i, j] for i in self.V + ["start"] if i != j and (k, i, j) in self.x
                            ]
                            if serve_terms:
                                serve_constrs[serve_key] = self.new_serve[k, c] >= gp.quicksum(serve_terms)

            # Add all serve constraints in batch if any exist
            if serve_constrs:
                self.model.addConstrs(serve_constrs, name="NewServe")

            # Prepare continuity penalties for objective
            continuity_expr = gp.quicksum(
                self.new_serve[k, c]
                * (day_continuity_penalty + continuity_penalty * (1 - self.is_historically_visited(k, c)))
                for k in self.K
                for c in self.C
                if (k, c) in self.new_serve
            )

            # Add continuity penalty term to objective
            objective_terms.append(continuity_expr)

        # Update objective function with all terms
        if len(objective_terms) > 1:  # If we added more terms beyond the base
            self.model.setObjective(sum(objective_terms), GRB.MINIMIZE)
            print("Updated objective function with penalties.")

        print("Model building complete.")
        return self.model
