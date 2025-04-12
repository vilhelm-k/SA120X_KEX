import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all optimization models"""

    # Initialization
    def __init__(
        self,
        caregivers,  # ID,ModeOfTransport,Attributes,EarliestStartTime,LatestEndTime,StartLocation,EndLocation
        tasks,  # ID,ClientID,StartTime,EndTime,TaskType,PlannedCaregiverID
        clients,  # ID,Requirements,x,y
        drive_time_matrix,  # ClientIDs. HQ is 0. Col is from, row is to.
        walk_time_matrix,
        bicycle_time_matrix,
        continuity,
        historical_visits=None,
        warm_start_schedule=None,
    ):
        """
        Initialize the base model with common data

        Args:
            caregivers (pd.DataFrame): DataFrame containing caregiver information
            tasks (pd.DataFrame): DataFrame containing task information
            clients (pd.DataFrame): DataFrame containing client information
            drive_time_matrix (pd.DataFrame): Matrix of travel times between locations for driving
            walk_time_matrix (pd.DataFrame): Matrix of travel times for walking
            bicycle_time_matrix (pd.DataFrame): Matrix of travel times for bicycling
            historical_visits (dict): Dictionary containing historical visits data
            warm_start_schedule (dict): Dictionary containing warm start schedule data
        """
        # Input data
        self.caregivers = caregivers
        self.tasks = tasks
        self.clients = clients
        self.drive_time_matrix = drive_time_matrix
        self.walk_time_matrix = walk_time_matrix
        self.bicycle_time_matrix = bicycle_time_matrix
        self.continuity = continuity
        self.historical_visits = historical_visits if historical_visits is not None else {}
        self.warm_start_schedule = warm_start_schedule

        # Preprocessed input data - these are small and used frequently, so keep as dictionaries
        self.K = self.caregivers.index.tolist()
        self.V = self.tasks.index.tolist()
        self.C = self.clients.index.tolist()
        self.s = {i: self.tasks.loc[i, "duration_minutes"] for i in self.V}  # s[i] Service time for i
        self.e = {i: self.tasks.loc[i, "start_minutes"] for i in self.V}  # e[i] Earliest start time for i
        self.l = {i: self.tasks.loc[i, "end_minutes"] for i in self.V}  # l[i] Latest end time for

        # Model variables
        self.model = None
        self.x = None
        self.t = None

        # Postprocessed results
        self.routes = None
        self.arrivals = None
        self.breaks = None

        # Save actual routes and arrival times from PlannedCaregiverID or warm_start_schedule
        self.real_routes = {}
        self.real_arrivals = {}

        if warm_start_schedule:
            self._preprocess_warm_start()
        else:
            self.real_routes, self.real_arrivals = self.__extract_real_routes_and_arrivals()

    def _preprocess_warm_start(self):
        """Preprocess warm start data to extract real routes and arrival times."""
        if not self.warm_start_schedule:
            return

        for caregiver_id, caregiver_data in self.warm_start_schedule.items():
            self.real_routes[caregiver_id] = []
            self.real_arrivals[caregiver_id] = {}

            for leg in caregiver_data.get("legs", []):
                from_id = leg["from"]["id"]
                to_id = leg["to"]["id"]
                self.real_routes[caregiver_id].append((from_id, to_id))

                # Store arrival times
                if "arrival" in leg["to"]:
                    self.real_arrivals[caregiver_id][to_id] = leg["to"]["arrival"]

            # Add start and end times if available
            if "start" in caregiver_data:
                self.real_arrivals[caregiver_id]["start"] = caregiver_data["start"]
            if "end" in caregiver_data:
                self.real_arrivals[caregiver_id]["end"] = caregiver_data["end"]

    def __extract_real_routes_and_arrivals(self):
        """
        Extract the real routes and arrival times based on the PlannedCaregiverID in tasks
        Returns:
            tuple: (real_routes, real_arrivals) dictionaries
        """
        # Initialize dictionaries for real routes and arrivals
        real_routes = {k: [] for k in self.K}
        real_arrivals = {k: {} for k in self.K}

        # Extract unique caregivers from tasks with PlannedCaregiverID
        if "PlannedCaregiverID" not in self.tasks.columns:
            return real_routes, real_arrivals

        # Get caregivers that have assigned tasks
        assigned_caregivers = set(self.tasks["PlannedCaregiverID"].dropna()).intersection(set(self.K))

        # Process each caregiver's route
        for k in assigned_caregivers:
            # Get all tasks for this caregiver sorted by start time
            caregiver_tasks = self.tasks[self.tasks["PlannedCaregiverID"] == k].sort_values("start_minutes")

            if caregiver_tasks.empty:
                continue

            # Set start and end times
            first_task = caregiver_tasks.iloc[0].name
            last_task = caregiver_tasks.iloc[-1].name

            # Calculate start and end times
            real_arrivals[k]["start"] = self.e[first_task] - self.c(k, "start", first_task)
            real_arrivals[k]["end"] = self.l[last_task] + self.c(k, last_task, "end")

            # Build the route
            prev_task = "start"
            for idx, task in caregiver_tasks.iterrows():
                # Add this leg of the route (prev_task -> current_task)
                real_routes[k].append((prev_task, idx))

                # Record arrival time (use the planned start time)
                real_arrivals[k][idx] = task["start_minutes"]

                # Update previous task
                prev_task = idx

            # Add the final leg back to end
            real_routes[k].append((prev_task, "end"))

        return real_routes, real_arrivals

    def is_historically_visited(self, caregiver_id, client_id):
        """Check if a caregiver has historically visited a client."""
        if not self.historical_visits:
            return False

        # Check if caregiver has visited client in historical data
        if caregiver_id in self.historical_visits:
            return client_id in self.historical_visits[caregiver_id]

        return False

    def c(self, caregiver_id, from_loc, to_loc):
        """Calculate travel time between locations based on caregiver's mode of transport."""
        if from_loc == "start" or to_loc == "end" or to_loc == "start" or from_loc == "end":
            return 0

        # Get caregiver's mode of transport
        if caregiver_id in self.caregivers.index:
            mode = self.caregivers.loc[caregiver_id, "ModeOfTransport"]

            # Get client IDs for the locations
            from_client = self.get_location(from_loc) if from_loc in self.V else from_loc
            to_client = self.get_location(to_loc) if to_loc in self.V else to_loc

            # Get travel time based on mode of transport
            if mode == "Car":
                if from_client in self.drive_time_matrix.index and to_client in self.drive_time_matrix.columns:
                    return self.drive_time_matrix.loc[from_client, to_client]
            elif mode == "Walk":
                if from_client in self.walk_time_matrix.index and to_client in self.walk_time_matrix.columns:
                    return self.walk_time_matrix.loc[from_client, to_client]
            elif mode == "Bicycle":
                if from_client in self.bicycle_time_matrix.index and to_client in self.bicycle_time_matrix.columns:
                    return self.bicycle_time_matrix.loc[from_client, to_client]

        # Default to a large value if travel time can't be determined
        return float("inf")

    def is_pair_feasible(self, k, i, j):
        """
        Check if a pair of tasks is feasible based on time constraints.
        """
        if i == j:
            return False

        # Determine which task comes first in time
        if self.e[i] < self.e[j]:
            first, last = i, j
        else:
            first, last = j, i

        # Check if there's enough time between the tasks
        return self.e[last] >= self.l[first] + self.c(k, first, last)

    def get_client_tasks(self, client_id):
        """Get all tasks for a client."""
        if client_id not in self.clients.index:
            return []

        # Filter tasks that belong to this client
        return self.tasks[self.tasks["ClientID"] == client_id].index.tolist()

    def is_caregiver_qualified(self, caregiver_id, task_id):
        """Check if a caregiver is qualified to perform a task."""
        if task_id == "start" or task_id == "end":
            return True

        # Get caregiver attributes and task requirements
        if caregiver_id in self.caregivers.index and task_id in self.tasks.index:
            caregiver_attrs = self.caregivers.loc[caregiver_id, "Attributes"]
            task_reqs = self.tasks.loc[task_id, "Requirements"]

            # Check if caregiver has all required attributes for the task
            return all(req in caregiver_attrs for req in task_reqs)

        return False  # If caregiver or task don't exist, return False

    def is_break_feasible(self, caregiver_id, task_i, task_j, break_length):
        """Check if a break is feasible between two tasks."""
        if task_i == "start" or task_j == "end":
            return False

        # Get task times
        if task_i in self.tasks.index and task_j in self.tasks.index:
            # Calculate end time of task i
            task_i_end = self.l[task_i] + self.s[task_i]

            # Calculate start time of task j
            task_j_start = self.e[task_j]

            # Calculate travel time between tasks
            travel_time = self.c(caregiver_id, task_i, task_j)

            # Check if there's enough time for a break between tasks
            return (task_j_start - task_i_end - travel_time) >= break_length

        return False  # If tasks don't exist, return False

    # Model building
    @abstractmethod
    def build(self):
        """Build the model. To be implemented by subclasses."""
        pass

    def optimize(self, **kwargs):
        """
        Optimize the model with given parameters

        Args:
            warm_start (bool): Whether to use warm start values from PlannedCaregiverID.
                              Setting this to True can significantly improve solve times
                              by starting from a feasible solution based on existing assignments.
            **kwargs: Optimization parameters like TimeLimit, MIPGap, etc.

        Returns:
            The optimized model
        """
        if self.model is None:
            raise ValueError("Model must be built before optimization")

        # Set any optimization parameters
        for param, value in kwargs.items():
            self.model.setParam(param, value)

        self.model.optimize()
        return self.model

    # Helper functions
    def get_location(self, task_id):
        """Get the location (client ID) for a task."""
        if task_id == "start" or task_id == "end":
            return None
        if task_id in self.V:
            return self.tasks.loc[task_id, "ClientID"]
        return None

    def get_endpoint(self, caregiver_id, endpoint):
        """
        Get the endpoint of a caregiver given its ID.

        Args:
            caregiver_id (int): The ID of the caregiver
            endpoint (str): The endpoint to get ("start" or "end")

        Returns:
            The endpoint of the caregiver.
        """
        if endpoint == "start":
            return self.caregivers.loc[caregiver_id, "StartLocation"]
        elif endpoint == "end":
            return self.caregivers.loc[caregiver_id, "EndLocation"]
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")

    # Postprocessing
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

    def _extract_breaks(self):
        """
        Extracts the break assignments for each caregiver.
        """
        breaks = {k: [] for k in self.K}

        # Iterate over all caregivers and tasks
        for k in self.K:
            for i in self.V:
                if self.B[k, i].X > 0.5:
                    breaks[k].append(i)

        self.breaks = breaks
        return breaks

    def _extract_arrival_times(self):
        """
        Extract the arrival times at each task for each caregiver.
        """
        self.arrivals = {}
        for k in self.K:
            self.arrivals[k] = {}
            self.arrivals[k]["start"] = self.S[k].X
            self.arrivals[k]["end"] = self.E[k].X

            for _, j in self.routes[k]:
                if j == "end":
                    continue
                self.arrivals[k][j] = self.e[j]
        return self.arrivals

    def get_solution(self):
        """
        Get the solution of the optimized model.
        To be implemented by subclasses based on their specific structure.
        """
        if self.model is None:
            raise ValueError("Model must be optimized before extracting the solution")
        if self.model.Status != GRB.OPTIMAL:
            print(f"Model not optimally solved. Status: {self.model.Status}")
        self._extract_routes()
        self._extract_arrival_times()
        if hasattr(self, "B"):
            self._extract_breaks()
        return self.routes, self.arrivals, self.breaks

    def distance(self, i, j):
        """Get distance between two locations based on driving time matrix."""
        if i == "start" or j == "end" or j == "start" or i == "end":
            return 0

        # If i and j are task IDs, get their client IDs
        client_i = self.get_location(i) if i in self.V else i
        client_j = self.get_location(j) if j in self.V else j

        # Return distance based on drive time matrix
        if client_i in self.drive_time_matrix.index and client_j in self.drive_time_matrix.columns:
            return self.drive_time_matrix.loc[client_i, client_j]
        return float("inf")

    def get_task_client(self, task_id):
        """Get client for a task."""
        if task_id == "start" or task_id == "end":
            return None
        if task_id in self.V:
            return self.tasks.loc[task_id, "ClientID"]
        return None

    def get_schedule(self):
        """Extract the schedule from the solved model.
        Returns a dictionary with keys for caregivers and values describing their routes.
        """
        if self.model is None or self.model.Status != GRB.OPTIMAL:
            raise ValueError("Model not solved optimally.")

        schedule = {}

        # Get variable dictionaries
        x_vars = self.model.getVars("x*")
        is_used_vars = self.model.getVars("is_used*")
        start_time_vars = self.model.getVars("T_start*")
        end_time_vars = self.model.getVars("T_end*")

        # Process x variables into a dictionary for easier access
        x_dict = {}
        for var in x_vars:
            if var.X > 0.5:  # Only consider variables with value close to 1
                # Parse variable name to get indices
                name_parts = var.VarName.split("[")
                if len(name_parts) < 2:
                    continue

                indices = name_parts[1].rstrip("]").split(",")
                if len(indices) != 3:
                    continue

                k, i, j = indices
                # Clean up any quotes
                k = k.strip("'")
                i = i.strip("'")
                j = j.strip("'")

                # Store in dictionary
                if k not in x_dict:
                    x_dict[k] = {}
                if i not in x_dict[k]:
                    x_dict[k][i] = {}
                x_dict[k][i][j] = var.X

        # Process is_used variables
        is_used_dict = {}
        for var in is_used_vars:
            name_parts = var.VarName.split("[")
            if len(name_parts) < 2:
                continue
            k = name_parts[1].rstrip("]").strip("'")
            is_used_dict[k] = var.X

        # Process start and end time variables
        start_time_dict = {}
        end_time_dict = {}
        for var in start_time_vars:
            name_parts = var.VarName.split("[")
            if len(name_parts) < 2:
                continue
            k = name_parts[1].rstrip("]").strip("'")
            start_time_dict[k] = var.X

        for var in end_time_vars:
            name_parts = var.VarName.split("[")
            if len(name_parts) < 2:
                continue
            k = name_parts[1].rstrip("]").strip("'")
            end_time_dict[k] = var.X

        # Get break variables if they exist
        break_dict = {}
        break_vars = self.model.getVars("B*")
        for var in break_vars:
            if var.X > 0.5:  # Only consider variables with value close to 1
                name_parts = var.VarName.split("[")
                if len(name_parts) < 2:
                    continue
                indices = name_parts[1].rstrip("]").split(",")
                if len(indices) != 2:
                    continue

                k, i = indices
                # Clean up any quotes
                k = k.strip("'")
                i = i.strip("'")

                if k not in break_dict:
                    break_dict[k] = []
                break_dict[k].append(i)

        # Get overtime variables if they exist
        overtime_dict = {}
        overtime_vars = self.model.getVars("overtime*")
        for var in overtime_vars:
            name_parts = var.VarName.split("[")
            if len(name_parts) < 2:
                continue
            k = name_parts[1].rstrip("]").strip("'")
            overtime_dict[k] = var.X

        # Process each caregiver
        for k in self.K:
            # Check if caregiver is used
            if k not in is_used_dict or is_used_dict[k] < 0.5:
                continue

            caregiver_schedule = {
                "id": k,
                "legs": [],
                "total_time": end_time_dict.get(k, 0) - start_time_dict.get(k, 0),
                "start": start_time_dict.get(k, 0),
                "end": end_time_dict.get(k, 0),
            }

            # Add overtime if present
            if k in overtime_dict:
                caregiver_schedule["overtime"] = overtime_dict[k]

            # Add breaks if present
            if k in break_dict:
                caregiver_schedule["breaks_after"] = break_dict[k]

            # Construct route
            route = []
            current = "start"

            # Follow the route from start to end
            while current != "end":
                if k not in x_dict or current not in x_dict[k]:
                    break  # No route for this caregiver or no next task

                # Find the next task
                next_task = None
                for j, val in x_dict[k][current].items():
                    if val > 0.5:
                        next_task = j
                        break

                if next_task is None:
                    break  # No next task found

                # Add leg to the route
                if current != "start":
                    from_task = {"id": current, "client": self.get_task_client(current)}
                else:
                    from_task = {"id": "start"}

                if next_task != "end":
                    to_task = {"id": next_task, "client": self.get_task_client(next_task)}

                    # Calculate arrival time (not exactly accurate but a reasonable estimate)
                    if current == "start":
                        # First task - from caregiver start time
                        arrival = start_time_dict.get(k, 0)
                    else:
                        # Later tasks - add processing and travel time
                        arrival = route[-1]["to"]["departure"] + self.distance(current, next_task)

                    # Add arrival and departure times
                    to_task["arrival"] = arrival
                    to_task["departure"] = arrival + self.s[next_task]
                else:
                    to_task = {"id": "end"}

                # Create leg
                leg = {
                    "from": from_task,
                    "to": to_task,
                }

                if current != "start" and next_task != "end":
                    leg["travel_time"] = self.distance(current, next_task)

                route.append(leg)
                current = next_task

            caregiver_schedule["legs"] = route
            schedule[k] = caregiver_schedule

        return schedule
