import pandas as pd
import numpy as np


def calculate_metrics(model):
    """
    Calculate comprehensive metrics for the home care scheduling solution.

    Parameters:
    - model: A solved optimization model instance

    Returns:
    - A dictionary containing both individual caregiver metrics and aggregate metrics
    """
    # Check if model has been solved
    if model.routes is None or model.arrivals is None:
        raise ValueError("Model must be solved before calculating metrics.")

    # Calculate individual caregiver metrics
    caregiver_metrics = calculate_caregiver_metrics(model)

    # Calculate aggregate metrics (now passing the model directly)
    aggregate_metrics = calculate_aggregate_metrics(caregiver_metrics, model)

    # Combine all metrics into a single dictionary
    return {"caregiver_metrics": caregiver_metrics, "aggregate_metrics": aggregate_metrics}


def calculate_caregiver_metrics(model):
    """
    Calculate metrics for each individual caregiver using a time-sequential approach.

    Parameters:
    - model: A solved optimization model instance

    Returns:
    - A dictionary with caregiver IDs as keys and their respective metrics as values
    """
    routes = model.routes
    arrivals = model.arrivals
    service_times = model.s
    tasks = model.tasks

    # Check if breaks exist in the model
    has_breaks = hasattr(model, "breaks") and model.breaks is not None

    caregiver_metrics = {}

    for k in routes:
        # Skip caregivers with no assigned tasks
        if not routes[k]:
            caregiver_metrics[k] = {
                "travel_time": 0,
                "service_time": 0,
                "waiting_time": 0,
                "break_time": 0,
                "total_time": 0,
                "utilization": 0,
                "number_of_tasks": 0,
                "temporal_violations": [],
                "proportions": {"travel": 0, "service": 0, "waiting": 0, "break": 0},
            }
            continue

        # Initialize metrics
        travel_time = 0
        service_time = 0
        waiting_time = 0
        break_time = 0
        task_count = 0
        temporal_violations = []

        # Use the same time-sequential approach as in visualize_schedule
        current_time = arrivals[k]["start"]

        # Process each route segment in order
        for idx, (i, j) in enumerate(routes[k]):
            if i != "start":
                # Get actual arrival time at task i
                task_arrival = arrivals[k][i]

                # Add waiting if the caregiver arrived earlier than the current time tracker
                if current_time < task_arrival:
                    waiting_duration = task_arrival - current_time
                    waiting_time += waiting_duration
                    current_time = task_arrival

                # Add service time for task i
                task_duration = service_times[i]
                service_time += task_duration
                task_count += 1
                current_time += task_duration

                # Add break time if this task is followed by a break
                if has_breaks and k in model.breaks and i in model.breaks[k]:
                    # Standard break duration (30 minutes by default)
                    break_duration = 30  # This could be adjusted based on model parameters
                    break_time += break_duration
                    current_time += break_duration

            # Check for temporal violations before adding travel time
            if j != "end" and i != "start":
                # Calculate the time left after completing task i
                departure_time = current_time
                # Calculate required travel time to j
                travel_required = model.c(k, i, j)
                # Calculate the earliest possible arrival at j
                earliest_arrival = departure_time + travel_required
                # Get the actual arrival time at j
                actual_arrival = arrivals[k][j]

                # Check if this violates the temporal constraint
                if earliest_arrival > actual_arrival:
                    violation_minutes = earliest_arrival - actual_arrival
                    temporal_violations.append(
                        {
                            "from_task": i,
                            "to_task": j,
                            "departure_time": departure_time,
                            "required_travel": travel_required,
                            "earliest_arrival": earliest_arrival,
                            "actual_arrival": actual_arrival,
                            "violation_minutes": violation_minutes,
                            "sequence": idx,
                        }
                    )

            # Add travel time to the next location
            travel_duration = model.c(k, i, j)
            travel_time += travel_duration
            current_time += travel_duration

        # Calculate total schedule time (from start to end)
        total_time = arrivals[k]["end"] - arrivals[k]["start"]

        # Calculate utilization (service time as percentage of total time)
        utilization = (service_time / total_time * 100) if total_time > 0 else 0

        # Calculate proportions
        proportions = {
            "travel": (travel_time / total_time * 100) if total_time > 0 else 0,
            "service": (service_time / total_time * 100) if total_time > 0 else 0,
            "waiting": (waiting_time / total_time * 100) if total_time > 0 else 0,
            "break": (break_time / total_time * 100) if total_time > 0 else 0,
        }

        # Verify that time accounting adds up
        accounted_time = service_time + travel_time + waiting_time + break_time
        time_diff = abs(total_time - accounted_time)
        if time_diff > 1:  # Tolerance of 1 minute due to potential floating point issues
            print(f"Warning: Caregiver {k} has a time accounting discrepancy of {time_diff:.2f} minutes")
            print(f"  Total time: {total_time:.2f}, Accounted time: {accounted_time:.2f}")
            print(
                f"  Service: {service_time:.2f}, Travel: {travel_time:.2f}, Waiting: {waiting_time:.2f}, Break: {break_time:.2f}"
            )

        # Store metrics for this caregiver
        caregiver_metrics[k] = {
            "travel_time": travel_time,
            "service_time": service_time,
            "waiting_time": waiting_time,
            "break_time": break_time,
            "total_time": total_time,
            "utilization": utilization,
            "number_of_tasks": task_count,
            "temporal_violations": temporal_violations,
            "proportions": proportions,
        }

    return caregiver_metrics


def calculate_aggregate_metrics(caregiver_metrics, model):
    """
    Calculate aggregate metrics across all caregivers and clients.

    Parameters:
    - caregiver_metrics: Dictionary of individual caregiver metrics
    - model: A solved optimization model instance for accessing continuity data

    Returns:
    - A dictionary containing aggregate metrics
    """
    # Extract values for all caregivers
    travel_times = [metrics["travel_time"] for metrics in caregiver_metrics.values()]
    service_times = [metrics["service_time"] for metrics in caregiver_metrics.values()]
    waiting_times = [metrics["waiting_time"] for metrics in caregiver_metrics.values()]
    break_times = [metrics["break_time"] for metrics in caregiver_metrics.values()]
    total_times = [metrics["total_time"] for metrics in caregiver_metrics.values()]
    utilizations = [metrics["utilization"] for metrics in caregiver_metrics.values()]
    task_counts = [metrics["number_of_tasks"] for metrics in caregiver_metrics.values()]

    # Collect all temporal violations
    all_violations = []
    for k, metrics in caregiver_metrics.items():
        for violation in metrics["temporal_violations"]:
            violation_copy = violation.copy()
            violation_copy["caregiver"] = k
            all_violations.append(violation_copy)

    # Calculate total values
    total_travel_time = sum(travel_times)
    total_service_time = sum(service_times)
    total_waiting_time = sum(waiting_times)
    total_break_time = sum(break_times)
    total_schedule_time = sum(total_times)
    total_tasks = sum(task_counts)
    total_violations = len(all_violations)
    total_violation_minutes = sum(v["violation_minutes"] for v in all_violations)

    # Calculate average values (only for caregivers with assigned tasks)
    active_caregivers = [k for k in caregiver_metrics if caregiver_metrics[k]["number_of_tasks"] > 0]
    num_active_caregivers = len(active_caregivers)

    avg_travel_time = total_travel_time / num_active_caregivers if num_active_caregivers > 0 else 0
    avg_service_time = total_service_time / num_active_caregivers if num_active_caregivers > 0 else 0
    avg_waiting_time = total_waiting_time / num_active_caregivers if num_active_caregivers > 0 else 0
    avg_break_time = total_break_time / num_active_caregivers if num_active_caregivers > 0 else 0
    avg_schedule_time = total_schedule_time / num_active_caregivers if num_active_caregivers > 0 else 0
    avg_utilization = total_service_time / total_schedule_time * 100 if total_schedule_time > 0 else 0
    avg_tasks_per_caregiver = total_tasks / num_active_caregivers if num_active_caregivers > 0 else 0
    avg_violations_per_caregiver = total_violations / num_active_caregivers if num_active_caregivers > 0 else 0

    # Count caregivers with violations
    caregivers_with_violations = sum(1 for k in caregiver_metrics if caregiver_metrics[k]["temporal_violations"])

    # Calculate global proportions
    global_proportions = {
        "travel": (total_travel_time / total_schedule_time * 100) if total_schedule_time > 0 else 0,
        "service": (total_service_time / total_schedule_time * 100) if total_schedule_time > 0 else 0,
        "waiting": (total_waiting_time / total_schedule_time * 100) if total_schedule_time > 0 else 0,
        "break": (total_break_time / total_schedule_time * 100) if total_schedule_time > 0 else 0,
    }

    # ---- Calculate continuity metrics ----
    continuity_metrics = calculate_continuity_metrics(model)

    # Compile aggregate metrics
    aggregate_metrics = {
        "total": {
            "travel_time": total_travel_time,
            "service_time": total_service_time,
            "waiting_time": total_waiting_time,
            "break_time": total_break_time,
            "schedule_time": total_schedule_time,
            "number_of_tasks": total_tasks,
            "violation_count": total_violations,
            "violation_minutes": total_violation_minutes,
        },
        "average": {
            "travel_time": avg_travel_time,
            "service_time": avg_service_time,
            "waiting_time": avg_waiting_time,
            "break_time": avg_break_time,
            "schedule_time": avg_schedule_time,
            "utilization": avg_utilization,
            "tasks_per_caregiver": avg_tasks_per_caregiver,
            "violations_per_caregiver": avg_violations_per_caregiver,
        },
        "proportions": global_proportions,
        "active_caregivers": num_active_caregivers,
        "total_caregivers": len(caregiver_metrics),
        "caregivers_with_violations": caregivers_with_violations,
        "all_violations": all_violations,
        "continuity": continuity_metrics,
    }

    return aggregate_metrics


def calculate_continuity_metrics(model):
    """
    Calculate continuity metrics for each client.

    Parameters:
    - model: A solved optimization model instance

    Returns:
    - A dictionary containing continuity metrics for each client and system-wide metrics
    """
    # Extract model data
    routes = model.routes
    H = model.H  # Historical continuity data
    C = model.C  # List of clients
    Vc = model.Vc  # Tasks for each client

    # Map each task to its assigned caregiver
    task_caregiver_map = {}
    for k, route in routes.items():
        for i, j in route:
            if i != "start":  # We only care about actual tasks
                task_caregiver_map[i] = k

    # Calculate client-specific continuity metrics
    client_continuity = {}
    for c in C:
        client_tasks = Vc[c]
        total_tasks = len(client_tasks)

        # Skip clients with no tasks
        if total_tasks == 0:
            continue

        # Identify caregivers assigned to this client's tasks
        caregivers_today = {}
        for task_id in client_tasks:
            if task_id in task_caregiver_map:
                k = task_caregiver_map[task_id]
                caregivers_today[k] = caregivers_today.get(k, 0) + 1

        # Calculate historical continuity
        historical_caregivers = [k for k in model.K if H[k, c] == 1]
        assigned_historical = [k for k in caregivers_today if H[k, c] == 1]

        unique_caregivers = len(caregivers_today)
        historical_tasks = sum(caregivers_today.get(k, 0) for k in historical_caregivers)

        # Calculate continuity score
        historical_continuity_score = 0
        if unique_caregivers > 0:
            historical_continuity_score = (len(assigned_historical) / unique_caregivers) * 100

        # Store metrics for this client
        client_continuity[c] = {
            "unique_caregivers": unique_caregivers,
            "total_tasks": total_tasks,
            "tasks_per_caregiver": total_tasks / unique_caregivers if unique_caregivers else 0,
            "historical_caregivers_count": len(historical_caregivers),
            "assigned_historical_caregivers": len(assigned_historical),
            "historical_continuity_score": historical_continuity_score,
            "historical_tasks": historical_tasks,
            "non_historical_tasks": total_tasks - historical_tasks,
        }

    # Calculate system-wide metrics using client data
    active_clients = len(client_continuity)

    # Count totals and calculate averages
    total_unique_caregivers = sum(metrics["unique_caregivers"] for metrics in client_continuity.values())
    total_historical_caregivers = sum(
        metrics["assigned_historical_caregivers"] for metrics in client_continuity.values()
    )
    total_tasks = sum(metrics["total_tasks"] for metrics in client_continuity.values())
    total_historical_tasks = sum(metrics["historical_tasks"] for metrics in client_continuity.values())

    # Count clients with perfect continuity
    perfect_continuity = sum(1 for metrics in client_continuity.values() if metrics["unique_caregivers"] == 1)
    perfect_historical = sum(
        1 for metrics in client_continuity.values() if metrics["historical_continuity_score"] == 100
    )

    system_continuity = {
        "avg_caregivers_per_client": total_unique_caregivers / active_clients,
        "avg_historical_continuity": (
            (total_historical_caregivers / total_unique_caregivers * 100) if total_unique_caregivers > 0 else 0
        ),
        "historical_task_percentage": (total_historical_tasks / total_tasks * 100) if total_tasks > 0 else 0,
        "total_historical_tasks": total_historical_tasks,
        "total_tasks": total_tasks,
        "perfect_continuity_clients": perfect_continuity,
        "perfect_historical_continuity_clients": perfect_historical,
    }

    return {"client_continuity": client_continuity, "system_continuity": system_continuity}
