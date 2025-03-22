import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
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

    for k in model.K:
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

    # Get y positions for each caregiver in reverse order
    y_positions = {cg: len(model.K) - 1 - i for i, cg in enumerate(model.K)}

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
            if width >= 40:
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


def visualize_metrics(model, display_mode="dashboard", dashboard_figsize=(20, 10)):
    """
    Create visualizations of the metrics from the model solution.

    Parameters:
    - model: A solved optimization model instance
    - display_mode: String, controls which figures to display
        - 'dashboard': Only display the dashboard (default)
        - 'individual': Only display the individual figures
        - 'all': Display both dashboard and individual figures
        - 'none': Don't display any figures, just return them
    - dashboard_figsize: Size of the dashboard figure

    Returns:
    - If display_mode is 'dashboard': The dashboard figure
    - If display_mode is 'individual': A tuple of individual figures (fig1, fig2, fig3, fig4, fig5)
    - If display_mode is 'all' or 'none': A tuple of all figures (fig1, fig2, fig3, fig4, fig5, dashboard_fig)
    """
    # Turn off interactive mode to prevent automatic display
    was_interactive = plt.isinteractive()
    if was_interactive:
        plt.ioff()
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

    # Create the dashboard figure
    dashboard_fig = plt.figure(figsize=dashboard_figsize)
    gs = gridspec.GridSpec(2, 3, figure=dashboard_fig)

    # We'll create the plots directly on the dashboard figure
    # 1. Time allocation bar chart (top left)
    dash_ax1 = dashboard_fig.add_subplot(gs[0, 0])
    dash_ax1.bar(index, service_times, label="Service", color="skyblue")
    dash_ax1.bar(index, travel_times, bottom=service_bottom, label="Travel", color="orange")
    dash_ax1.bar(index, waiting_times, bottom=service_travel_bottom, label="Waiting", color="lightgray")
    dash_ax1.set_xticks(index)
    dash_ax1.set_xticklabels(caregivers)
    dash_ax1.set_xlabel("Caregiver")
    dash_ax1.set_ylabel("Time (minutes)")
    dash_ax1.set_title("Time Allocation by Caregiver")
    dash_ax1.legend()

    # 2. Pie chart (top right)
    dash_ax2 = dashboard_fig.add_subplot(gs[0, 2])
    dash_ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    dash_ax2.axis("equal")
    dash_ax2.set_title("Overall Time Allocation")

    # 3. Utilization bar chart (middle left)
    dash_ax3 = dashboard_fig.add_subplot(gs[1, 0])
    dash_ax3.bar(index, utilizations, color="skyblue")
    dash_ax3.set_xticks(index)
    dash_ax3.set_xticklabels(caregivers)
    dash_ax3.axhline(y=avg_util, linestyle="--", color="red", label=f"Avg: {avg_util:.1f}%")
    dash_ax3.set_xlabel("Caregiver")
    dash_ax3.set_ylabel("Utilization (%)")
    dash_ax3.set_title("Caregiver Utilization")
    dash_ax3.set_ylim(0, 100)
    dash_ax3.legend()

    # 4. Tasks per caregiver (middle right)
    dash_ax4 = dashboard_fig.add_subplot(gs[1, 1])
    dash_ax4.bar(index, tasks_per_caregiver, color="skyblue")
    dash_ax4.set_xticks(index)
    dash_ax4.set_xticklabels(caregivers)
    dash_ax4.axhline(y=avg_tasks, linestyle="--", color="red", label=f"Avg: {avg_tasks:.1f}")
    dash_ax4.set_xlabel("Caregiver")
    dash_ax4.set_ylabel("Number of Tasks")
    dash_ax4.set_title("Tasks Assigned per Caregiver")
    dash_ax4.legend()

    # 5. Time proportions (bottom left)
    dash_ax5 = dashboard_fig.add_subplot(gs[0, 1])
    dash_ax5.bar(index, proportions_service, bar_width, label="Service", color="skyblue")
    dash_ax5.bar(index, proportions_travel, bar_width, bottom=proportions_service, label="Travel", color="orange")
    dash_ax5.bar(
        index, proportions_waiting, bar_width, bottom=bottom_service_travel, label="Waiting", color="lightgray"
    )
    dash_ax5.set_ylabel("Percentage (%)")
    dash_ax5.set_xlabel("Caregiver")
    dash_ax5.set_title("Time Allocation Proportions")
    dash_ax5.set_ylim(0, 100)
    dash_ax5.set_xticks(index)
    dash_ax5.set_xticklabels(caregivers)
    dash_ax5.legend()

    # 6. Summary statistics text box (bottom right)
    dash_ax6 = dashboard_fig.add_subplot(gs[1, 2])
    dash_ax6.axis("off")

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
    dash_ax6.text(
        0.05,
        0.95,
        "\n".join(summary_text),
        transform=dash_ax6.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()

    # Handle display based on display_mode
    individual_figs = [fig1, fig2, fig3, fig4, fig5]
    all_figs = individual_figs + [dashboard_fig]

    # Close figures we don't want to display
    if display_mode == "dashboard":
        for fig in individual_figs:
            plt.close(fig)
        # Only display the dashboard
        if was_interactive:
            plt.ion()  # Restore interactive mode
            dashboard_fig.show()
        return dashboard_fig

    elif display_mode == "individual":
        plt.close(dashboard_fig)
        # Only display individual figures
        if was_interactive:
            plt.ion()  # Restore interactive mode
            for fig in individual_figs:
                fig.show()
        return tuple(individual_figs)

    elif display_mode == "all":
        # Display all figures
        if was_interactive:
            plt.ion()  # Restore interactive mode
            for fig in all_figs:
                fig.show()
        return tuple(all_figs)

    else:  # 'none' or any other value
        # Don't display any figures, just return them
        for fig in all_figs:
            plt.close(fig)
        if was_interactive:
            plt.ion()  # Restore interactive mode
        return tuple(all_figs)


def visualize_routes(model, caregiver_ids=None, subplot_mode=False, figsize=None, dpi=100):
    """
    Visualize routes taken by selected caregivers with a simple, clean layout.

    Args:
        model: The optimized BaseModel or FlexibleModel instance with solution
        caregiver_ids: List of caregiver IDs to display (if None, all caregivers are shown)
        figsize: Size of the figure to create
        dpi: Resolution of the figure
        subplot_mode: If True, create a separate subplot for each caregiver;
                     if False, plot all routes on the same axes (default)

    Returns:
        If subplot_mode is False:
            matplotlib figure and single axis object
        If subplot_mode is True:
            matplotlib figure and array of axis objects (one per caregiver)
    """

    # Use model.clients
    clients_df = model.clients

    # Extract solution if not already done
    if model.routes is None or model.arrivals is None:
        raise ValueError("Model must be solved before visualization.")

    # If caregiver_ids is None, use all caregivers
    if caregiver_ids is None:
        caregiver_ids = model.K

    if figsize is None:
        figsize = (10, 2 * len(caregiver_ids))

    # Set up colors for caregivers - use distinct colors
    colors = list(mcolors.TABLEAU_COLORS)

    # Get active clients
    active_clients = model.tasks["ClientID"].unique()
    active_client_data = clients_df.loc[clients_df.index.isin(active_clients)]

    # Helper function to plot a caregiver's route
    def plot_caregiver_route(ax, k, k_idx):
        color = colors[k_idx % len(colors)]

        # Skip caregivers with no routes
        if not model.routes[k]:
            return False

        # Process route data
        route_x = []
        route_y = []
        route_tasks = []

        # Handle start point
        start_loc = model.get_endpoint(k, "start")
        if start_loc == "HQ":
            start_x, start_y = 0, 0
            route_x.append(start_x)
            route_y.append(start_y)
            route_tasks.append(None)

        # Process each task in the route
        for i, j in model.routes[k]:
            if j != "end":
                client_id = model.get_location(j)
                client_x = clients_df.loc[client_id, "x"]
                client_y = clients_df.loc[client_id, "y"]
                route_x.append(client_x)
                route_y.append(client_y)
                route_tasks.append(j)

        # Handle end point
        end_loc = model.get_endpoint(k, "end")
        if end_loc == "HQ":
            end_x, end_y = 0, 0
            route_x.append(end_x)
            route_y.append(end_y)
            route_tasks.append(None)

        # Skip if no route was created
        if len(route_x) <= 1:
            return False

        # Plot the route
        ax.plot(route_x, route_y, color=color, linewidth=2, alpha=0.7, label=f"Caregiver {k}")

        # Add order numbers for each visit
        visit_order = 1
        for i in range(len(route_x)):
            if route_tasks[i] is not None:
                ax.text(
                    route_x[i],
                    route_y[i],
                    str(visit_order),
                    fontsize=9,
                    color="black",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="circle", facecolor="white", edgecolor=color, linewidth=1.5, alpha=0.9),
                    zorder=3,
                )
                visit_order += 1

        return True

    if subplot_mode:
        # Create a figure with subplots for each caregiver
        n_caregivers = len(caregiver_ids)
        n_cols = min(2, n_caregivers)  # Use max 2 columns
        n_rows = int(np.ceil(n_caregivers / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)

        # Handle the case of a single caregiver (axes won't be an array)
        if n_caregivers == 1:
            axes = np.array([axes])

        # Flatten axes array for easier iteration
        axes = np.array(axes).flatten()

        # For each caregiver, create a separate subplot
        for k_idx, k in enumerate(caregiver_ids):
            ax = axes[k_idx]

            # Plot all client nodes in the background
            ax.scatter(active_client_data["x"], active_client_data["y"], color="lightgray", s=40, alpha=0.5)

            # Plot the caregiver's route
            has_route = plot_caregiver_route(ax, k, k_idx)

            # Plot HQ
            ax.scatter(0, 0, color="black", s=80, marker="s")

            # Set title and labels
            ax.set_title(f"Caregiver {k}" + (" (No Route)" if not has_route else ""))
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.3)

            # Adjust axes to have some margin
            ax.margins(0.1)

        # Hide any unused subplots
        for idx in range(len(caregiver_ids), len(axes)):
            axes[idx].set_visible(False)

    else:
        # Original single plot implementation
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Plot all client nodes in the background
        ax.scatter(active_client_data["x"], active_client_data["y"], color="lightgray", s=40, alpha=0.5)

        # For each caregiver, plot their route
        for k_idx, k in enumerate(caregiver_ids):
            plot_caregiver_route(ax, k, k_idx)

        # Plot HQ
        ax.scatter(0, 0, color="black", s=80, marker="s")

        # Add legend for caregivers
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Set axis labels
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

        # Adjust axes to have some margin
        ax.margins(0.1)

    plt.tight_layout()
    return fig, ax if not subplot_mode else (fig, axes)
