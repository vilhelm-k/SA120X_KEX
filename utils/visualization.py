import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from utils.metrics import calculate_metrics


def visualize_schedule(model):
    """
    Create a home care schedule visualization using Matplotlib.
    Shows each caregiver's route with indices for tasks.

    Parameters:
    - model: A solved optimization model instance

    Returns:
    - fig: Matplotlib figure object
    """
    # Check if model has been solved
    if model.routes is None or model.arrivals is None:
        raise ValueError("Model must be solved before visualization.")

    # Extract data from the model
    routes = model.routes
    arrivals = model.arrivals
    travel_times = model.c
    service_times = model.s

    # Create schedule data for visualization
    schedule_data = []

    for k in routes:
        if not routes[k]:  # Skip empty routes
            continue

        current_time = arrivals[k]["start"]
        for index, (i, j) in enumerate(routes[k]):
            if i != "start":
                # Get actual arrival time
                task_arrival = arrivals[k][i]

                # Add waiting if arriving before task start
                if current_time < task_arrival:
                    waiting_duration = task_arrival - current_time
                    schedule_data.append(
                        {
                            "Caregiver": k,
                            "Activity": "Waiting",
                            "Index": None,
                            "Start": current_time,
                            "Duration": waiting_duration,
                            "Color": "lightgray",
                        }
                    )
                    current_time = task_arrival

                # Activity
                task_duration = service_times[i]
                schedule_data.append(
                    {
                        "Caregiver": k,
                        "Activity": f"Task {i}",
                        "Index": index,
                        "Start": current_time,
                        "Duration": task_duration,
                        "Color": "skyblue",
                    }
                )
                current_time += task_duration

            # Travel
            travel_duration = travel_times[k, i, j]
            schedule_data.append(
                {
                    "Caregiver": k,
                    "Activity": "Travel",
                    "Index": index,
                    "Start": current_time,
                    "Duration": travel_duration,
                    "Color": "orange",
                }
            )
            current_time += travel_duration

    # Create the visualization
    df = pd.DataFrame(schedule_data)

    if df.empty:
        print("No schedule data to visualize.")
        return None

    # Calculate figure dimensions
    num_caregivers = len([k for k in routes if routes[k]])
    fig_height = max(8, num_caregivers * 1.5)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, fig_height))

    # Get y positions for each caregiver
    caregivers_list = sorted(df["Caregiver"].unique())
    y_positions = {cg: i for i, cg in enumerate(caregivers_list)}

    # Calculate time range
    min_time = df["Start"].min()
    max_time = (df["Start"] + df["Duration"]).max()

    # Plot each activity as a colored rectangle
    for _, row in df.iterrows():
        y_pos = y_positions[row["Caregiver"]]
        x_start = row["Start"]
        width = row["Duration"]

        rect = patches.Rectangle(
            (x_start, y_pos - 0.4), width, 0.8, linewidth=1, edgecolor="black", facecolor=row["Color"], alpha=0.7
        )
        ax.add_patch(rect)

        # Add text labels for tasks
        if "Task" in row["Activity"]:
            # Determine what text to display based on width
            if width >= 25:
                # If box is wide enough, show task info
                task_id = row["Activity"].split(" ")[1]
                label = f"{task_id}\n({int(row['Index'])})"
            else:
                # For narrow boxes, just show index
                label = f"{int(row['Index'])}"

            ax.text(
                x_start + width / 2,
                y_pos,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    # Set up the axis
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"Caregiver {cg}" for cg in y_positions.keys()])
    ax.set_ylim(-0.5, len(y_positions) - 0.5)

    # Set x-axis to show hours
    max_hours = np.ceil(max_time / 60)
    hour_ticks = np.arange(0, max_hours * 60 + 1, 60)
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([f"{int(h/60)}:00" for h in hour_ticks])
    ax.set_xlim(min_time - 15, max_time + 15)

    # Add grid, title and labels
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)
    ax.set_title("Home Care Scheduling Solution", fontsize=16)
    ax.set_xlabel("Time (hours)", fontsize=12)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor="skyblue", edgecolor="black", alpha=0.7, label="Task"),
        patches.Patch(facecolor="orange", edgecolor="black", alpha=0.7, label="Travel"),
        patches.Patch(facecolor="lightgray", edgecolor="black", alpha=0.7, label="Waiting"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig


import matplotlib.pyplot as plt
import numpy as np


def visualize_metrics(model):
    """
    Create visualizations of the metrics from the model solution.
    This function creates multiple plots showing different aspects of the solution.

    Parameters:
    - model: A solved optimization model instance

    Returns:
    - A tuple of matplotlib figures
    """
    metrics = calculate_metrics(model)
    caregiver_metrics = metrics["caregiver_metrics"]
    aggregate_metrics = metrics["aggregate_metrics"]

    # Only include caregivers with assigned tasks
    active_caregivers = {k: v for k, v in caregiver_metrics.items() if v["number_of_tasks"] > 0}

    # Create figures only if there are active caregivers
    if not active_caregivers:
        print("No active caregivers to visualize.")
        return None

    # Get list of caregivers and use consistent indices
    caregivers = list(active_caregivers.keys())
    index = np.arange(len(caregivers))  # Use numeric indices for x-axis positions

    # 1. Time allocation bar chart
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Prepare data for stacked bar chart
    service_times = [active_caregivers[k]["service_time"] for k in caregivers]
    travel_times = [active_caregivers[k]["travel_time"] for k in caregivers]
    waiting_times = [active_caregivers[k]["waiting_time"] for k in caregivers]

    # Create stacked bar chart using index positions
    ax1.bar(index, service_times, label="Service Time", color="skyblue")

    # Calculate bottom positions for stacking
    service_bottom = service_times
    ax1.bar(index, travel_times, bottom=service_bottom, label="Travel Time", color="orange")

    # Calculate bottom for waiting times
    service_travel_bottom = [s + t for s, t in zip(service_times, travel_times)]
    ax1.bar(index, waiting_times, bottom=service_travel_bottom, label="Waiting Time", color="lightgray")

    # Add total time as text on top of each bar
    for i, k in enumerate(caregivers):
        total_time = active_caregivers[k]["total_time"]
        ax1.text(
            i, sum([service_times[i], travel_times[i], waiting_times[i]]) + 5, f"{total_time:.0f} min", ha="center"
        )

    # Set x-ticks to show actual caregiver IDs
    ax1.set_xticks(index)
    ax1.set_xticklabels(caregivers)

    ax1.set_xlabel("Caregiver")
    ax1.set_ylabel("Time (minutes)")
    ax1.set_title("Time Allocation by Caregiver")
    ax1.legend()

    # 2. Utilization chart
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    utilizations = [active_caregivers[k]["utilization"] for k in caregivers]

    # Use index positions for x-axis
    ax2.bar(index, utilizations, color="skyblue")

    # Set x-ticks to show actual caregiver IDs
    ax2.set_xticks(index)
    ax2.set_xticklabels(caregivers)

    ax2.set_xlabel("Caregiver")
    ax2.set_ylabel("Utilization (%)")
    ax2.set_title("Caregiver Utilization")
    ax2.set_ylim(0, 100)

    # Add average utilization line
    avg_util = aggregate_metrics["average"]["utilization"]
    ax2.axhline(y=avg_util, linestyle="--", color="red", label=f"Average: {avg_util:.1f}%")
    ax2.legend()

    # 3. Pie chart of overall time allocation
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    labels = ["Service Time", "Travel Time", "Waiting Time"]
    sizes = [
        aggregate_metrics["total"]["service_time"],
        aggregate_metrics["total"]["travel_time"],
        aggregate_metrics["total"]["waiting_time"],
    ]
    colors = ["skyblue", "orange", "lightgray"]

    # Calculate percentages for labels
    total = sum(sizes)
    percentages = [100 * size / total if total > 0 else 0 for size in sizes]
    label_texts = [f"{l} ({p:.1f}%)" for l, p in zip(labels, percentages)]

    # Create the pie chart
    wedges, texts, autotexts = ax3.pie(
        sizes,
        labels=label_texts,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        shadow=True,
        explode=(0.05, 0, 0),  # Slightly explode the service time slice
    )

    # Improve text legibility
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax3.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
    ax3.set_title("Overall Time Allocation")

    # 4. Tasks per caregiver
    fig4, ax4 = plt.subplots(figsize=(12, 6))

    tasks_per_caregiver = [active_caregivers[k]["number_of_tasks"] for k in caregivers]

    # Use index positions for x-axis
    ax4.bar(index, tasks_per_caregiver, color="skyblue")

    # Set x-ticks to show actual caregiver IDs
    ax4.set_xticks(index)
    ax4.set_xticklabels(caregivers)

    ax4.set_xlabel("Caregiver")
    ax4.set_ylabel("Number of Tasks")
    ax4.set_title("Tasks Assigned per Caregiver")

    # Add average line
    avg_tasks = aggregate_metrics["average"]["tasks_per_caregiver"]
    ax4.axhline(y=avg_tasks, linestyle="--", color="red", label=f"Average: {avg_tasks:.1f}")
    ax4.legend()

    # 5. Time proportions per caregiver
    fig5, ax5 = plt.subplots(figsize=(14, 6))

    proportions_service = [active_caregivers[k]["proportions"]["service"] for k in caregivers]
    proportions_travel = [active_caregivers[k]["proportions"]["travel"] for k in caregivers]
    proportions_waiting = [active_caregivers[k]["proportions"]["waiting"] for k in caregivers]

    # Set width of bars
    bar_width = 0.8

    # Create stacked percentage bars
    ax5.bar(index, proportions_service, bar_width, label="Service Time", color="skyblue")
    ax5.bar(index, proportions_travel, bar_width, bottom=proportions_service, label="Travel Time", color="orange")

    bottom_service_travel = [s + t for s, t in zip(proportions_service, proportions_travel)]
    ax5.bar(
        index, proportions_waiting, bar_width, bottom=bottom_service_travel, label="Waiting Time", color="lightgray"
    )

    # Add percentages as text
    for i in range(len(caregivers)):
        # Add service percentage
        if proportions_service[i] > 5:  # Only add text if slice is large enough
            ax5.text(
                i,
                proportions_service[i] / 2,
                f"{proportions_service[i]:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                color="black",
            )

        # Add travel percentage
        if proportions_travel[i] > 5:
            ax5.text(
                i,
                proportions_service[i] + proportions_travel[i] / 2,
                f"{proportions_travel[i]:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                color="black",
            )

        # Add waiting percentage
        if proportions_waiting[i] > 5:
            ax5.text(
                i,
                bottom_service_travel[i] + proportions_waiting[i] / 2,
                f"{proportions_waiting[i]:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                color="black",
            )

    ax5.set_ylabel("Percentage of Total Time (%)")
    ax5.set_xlabel("Caregiver")
    ax5.set_title("Time Allocation Proportions by Caregiver")
    ax5.set_yticks(range(0, 101, 10))
    ax5.set_xticks(index)
    ax5.set_xticklabels(caregivers)
    ax5.legend(loc="upper right")

    # Set the upper limit of the y-axis to 100%
    ax5.set_ylim(0, 100)

    # Add gridlines for readability
    ax5.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    return (fig1, fig2, fig3, fig4, fig5)


def create_summary_dashboard(model, figsize=(18, 10)):
    """
    Create a comprehensive dashboard combining key metrics and visualizations.

    Parameters:
    - model: A solved optimization model instance
    - figsize: Size of the dashboard figure

    Returns:
    - A matplotlib figure
    """
    import matplotlib.gridspec as gridspec

    # Calculate metrics
    metrics = calculate_metrics(model)
    caregiver_metrics = metrics["caregiver_metrics"]
    aggregate_metrics = metrics["aggregate_metrics"]

    # Only include caregivers with assigned tasks
    active_caregivers = {k: v for k, v in caregiver_metrics.items() if v["number_of_tasks"] > 0}

    if not active_caregivers:
        print("No active caregivers to visualize.")
        return None

    # Get list of caregivers and use consistent indices
    caregivers = list(active_caregivers.keys())
    index = np.arange(len(caregivers))  # Use numeric indices for x-axis positions

    # Create the dashboard figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])

    # 1. Time allocation bar chart
    ax1 = fig.add_subplot(gs[0, 0:2])

    service_times = [active_caregivers[k]["service_time"] for k in caregivers]
    travel_times = [active_caregivers[k]["travel_time"] for k in caregivers]
    waiting_times = [active_caregivers[k]["waiting_time"] for k in caregivers]

    # Use index positions for consistent x-axis
    ax1.bar(index, service_times, label="Service", color="skyblue")
    service_bottom = service_times
    ax1.bar(index, travel_times, bottom=service_bottom, label="Travel", color="orange")
    service_travel_bottom = [s + t for s, t in zip(service_times, travel_times)]
    ax1.bar(index, waiting_times, bottom=service_travel_bottom, label="Waiting", color="lightgray")

    # Set x-ticks to show actual caregiver IDs
    ax1.set_xticks(index)
    ax1.set_xticklabels(caregivers)

    ax1.set_xlabel("Caregiver")
    ax1.set_ylabel("Time (minutes)")
    ax1.set_title("Time Allocation by Caregiver")
    ax1.legend()

    # 2. Pie chart
    ax2 = fig.add_subplot(gs[0, 2])

    sizes = [
        aggregate_metrics["total"]["service_time"],
        aggregate_metrics["total"]["travel_time"],
        aggregate_metrics["total"]["waiting_time"],
    ]
    labels = ["Service", "Travel", "Waiting"]
    colors = ["skyblue", "orange", "lightgray"]

    ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax2.axis("equal")
    ax2.set_title("Overall Time Allocation")

    # 3. Utilization bar chart
    ax3 = fig.add_subplot(gs[1, 0])

    utilizations = [active_caregivers[k]["utilization"] for k in caregivers]

    # Use index positions for consistent x-axis
    ax3.bar(index, utilizations, color="skyblue")

    # Set x-ticks to show actual caregiver IDs
    ax3.set_xticks(index)
    ax3.set_xticklabels(caregivers)

    avg_util = aggregate_metrics["average"]["utilization"]
    ax3.axhline(y=avg_util, linestyle="--", color="red", label=f"Avg: {avg_util:.1f}%")

    ax3.set_xlabel("Caregiver")
    ax3.set_ylabel("Utilization (%)")
    ax3.set_title("Caregiver Utilization")
    ax3.set_ylim(0, 100)
    ax3.legend()

    # 4. Tasks per caregiver
    ax4 = fig.add_subplot(gs[1, 1])

    tasks_per_caregiver = [active_caregivers[k]["number_of_tasks"] for k in caregivers]

    # Use index positions for consistent x-axis
    ax4.bar(index, tasks_per_caregiver, color="skyblue")

    # Set x-ticks to show actual caregiver IDs
    ax4.set_xticks(index)
    ax4.set_xticklabels(caregivers)

    avg_tasks = aggregate_metrics["average"]["tasks_per_caregiver"]
    ax4.axhline(y=avg_tasks, linestyle="--", color="red", label=f"Avg: {avg_tasks:.1f}")

    ax4.set_xlabel("Caregiver")
    ax4.set_ylabel("Number of Tasks")
    ax4.set_title("Tasks Assigned per Caregiver")
    ax4.legend()

    # 5. Summary statistics text box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    agg = aggregate_metrics

    summary_text = [
        f"SCHEDULE SUMMARY",
        f"",
        f"Total tasks: {agg['total']['number_of_tasks']}",
        f"Active caregivers: {agg['active_caregivers']}/{agg['total_caregivers']}",
        f"",
        f"TIME ALLOCATION:",
        f"Service: {agg['total']['service_time']:.0f} min ({agg['proportions']['service']:.1f}%)",
        f"Travel: {agg['total']['travel_time']:.0f} min ({agg['proportions']['travel']:.1f}%)",
        f"Waiting: {agg['total']['waiting_time']:.0f} min ({agg['proportions']['waiting']:.1f}%)",
        f"Total time: {agg['total']['schedule_time']:.0f} min",
        f"",
        f"AVERAGES PER CAREGIVER:",
        f"Tasks: {agg['average']['tasks_per_caregiver']:.1f}",
        f"Service time: {agg['average']['service_time']:.0f} min",
        f"Travel time: {agg['average']['travel_time']:.0f} min",
        f"Waiting time: {agg['average']['waiting_time']:.0f} min",
        f"Utilization: {agg['average']['utilization']:.1f}%",
    ]

    # Add text box
    props = dict(boxstyle="round", facecolor="white", alpha=0.9)
    ax5.text(
        0.05, 0.95, "\n".join(summary_text), transform=ax5.transAxes, fontsize=11, verticalalignment="top", bbox=props
    )

    plt.tight_layout()
    return fig
