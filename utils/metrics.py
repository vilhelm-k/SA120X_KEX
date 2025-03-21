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

    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(caregiver_metrics)

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
    travel_times = model.c
    service_times = model.s
    tasks = model.tasks

    caregiver_metrics = {}

    for k in routes:
        # Skip caregivers with no assigned tasks
        if not routes[k]:
            caregiver_metrics[k] = {
                "travel_time": 0,
                "service_time": 0,
                "waiting_time": 0,
                "total_time": 0,
                "utilization": 0,
                "number_of_tasks": 0,
                "proportions": {"travel": 0, "service": 0, "waiting": 0},
            }
            continue

        # Initialize metrics
        travel_time = 0
        service_time = 0
        waiting_time = 0
        task_count = 0

        # Use the same time-sequential approach as in visualize_schedule
        current_time = arrivals[k]["start"]

        # Process each route segment in order
        for i, j in routes[k]:
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

            # Add travel time to the next location
            travel_duration = travel_times[k, i, j]
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
        }

        # Verify that time accounting adds up
        accounted_time = service_time + travel_time + waiting_time
        time_diff = abs(total_time - accounted_time)
        if time_diff > 1:  # Tolerance of 1 minute due to potential floating point issues
            print(f"Warning: Caregiver {k} has a time accounting discrepancy of {time_diff:.2f} minutes")
            print(f"  Total time: {total_time:.2f}, Accounted time: {accounted_time:.2f}")
            print(f"  Service: {service_time:.2f}, Travel: {travel_time:.2f}, Waiting: {waiting_time:.2f}")

        # Store metrics for this caregiver
        caregiver_metrics[k] = {
            "travel_time": travel_time,
            "service_time": service_time,
            "waiting_time": waiting_time,
            "total_time": total_time,
            "utilization": utilization,
            "number_of_tasks": task_count,
            "proportions": proportions,
        }

    return caregiver_metrics


def calculate_aggregate_metrics(caregiver_metrics):
    """
    Calculate aggregate metrics across all caregivers.

    Parameters:
    - caregiver_metrics: Dictionary of individual caregiver metrics

    Returns:
    - A dictionary containing aggregate metrics
    """
    # Extract values for all caregivers
    travel_times = [metrics["travel_time"] for metrics in caregiver_metrics.values()]
    service_times = [metrics["service_time"] for metrics in caregiver_metrics.values()]
    waiting_times = [metrics["waiting_time"] for metrics in caregiver_metrics.values()]
    total_times = [metrics["total_time"] for metrics in caregiver_metrics.values()]
    utilizations = [metrics["utilization"] for metrics in caregiver_metrics.values()]
    task_counts = [metrics["number_of_tasks"] for metrics in caregiver_metrics.values()]

    # Calculate total values
    total_travel_time = sum(travel_times)
    total_service_time = sum(service_times)
    total_waiting_time = sum(waiting_times)
    total_schedule_time = sum(total_times)
    total_tasks = sum(task_counts)

    # Calculate average values (only for caregivers with assigned tasks)
    active_caregivers = [k for k in caregiver_metrics if caregiver_metrics[k]["number_of_tasks"] > 0]
    num_active_caregivers = len(active_caregivers)

    avg_travel_time = total_travel_time / num_active_caregivers if num_active_caregivers > 0 else 0
    avg_service_time = total_service_time / num_active_caregivers if num_active_caregivers > 0 else 0
    avg_waiting_time = total_waiting_time / num_active_caregivers if num_active_caregivers > 0 else 0
    avg_schedule_time = total_schedule_time / num_active_caregivers if num_active_caregivers > 0 else 0
    avg_utilization = total_service_time / total_schedule_time * 100 if total_schedule_time > 0 else 0
    avg_tasks_per_caregiver = total_tasks / num_active_caregivers if num_active_caregivers > 0 else 0

    # Calculate global proportions
    global_proportions = {
        "travel": (total_travel_time / total_schedule_time * 100) if total_schedule_time > 0 else 0,
        "service": (total_service_time / total_schedule_time * 100) if total_schedule_time > 0 else 0,
        "waiting": (total_waiting_time / total_schedule_time * 100) if total_schedule_time > 0 else 0,
    }

    # Compile aggregate metrics
    aggregate_metrics = {
        "total": {
            "travel_time": total_travel_time,
            "service_time": total_service_time,
            "waiting_time": total_waiting_time,
            "schedule_time": total_schedule_time,
            "number_of_tasks": total_tasks,
        },
        "average": {
            "travel_time": avg_travel_time,
            "service_time": avg_service_time,
            "waiting_time": avg_waiting_time,
            "schedule_time": avg_schedule_time,
            "utilization": avg_utilization,
            "tasks_per_caregiver": avg_tasks_per_caregiver,
        },
        "proportions": global_proportions,
        "active_caregivers": num_active_caregivers,
        "total_caregivers": len(caregiver_metrics),
    }

    return aggregate_metrics
