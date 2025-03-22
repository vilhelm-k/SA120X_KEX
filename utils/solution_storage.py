import json
import pickle
import datetime
from pathlib import Path


def save_solution(model, name=None, format="json"):
    """
    Save a model's routes and arrival times to a file.

    Args:
        model: An instance of a model (subclass of BaseModel)
        name (str, optional): Name to identify this solution. If None, a timestamp will be used.
        format (str, optional): Format to save the solution in. Default is "json".
                                Options: "json", "pickle"

    Returns:
        str: Path to the saved solution file
    """
    # Create the directory if it doesn't exist
    save_dir = Path("data/saved_solutions")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp and optional name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if name:
        # Clean name to be filesystem-friendly
        name = "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in name)
        filename = f"{timestamp}_{name}"
    else:
        filename = timestamp

    # Ensure model solution is computed
    if model.routes is None or model.arrivals is None:
        model.get_solution()

    # Extract only the essential solution components
    solution_data = {
        "metadata": {
            "saved_at": timestamp,
            "name": name,
            "objective_value": model.model.objVal if hasattr(model.model, "objVal") else None,
            "model_status": model.model.Status,
        },
        "routes": {},
        "arrivals": {},
    }

    # Store routes (convert tuples to lists for JSON serialization)
    for k, route in model.routes.items():
        solution_data["routes"][str(k)] = [[str(i), str(j)] for i, j in route]

    # Store arrival times
    for k, arrivals in model.arrivals.items():
        solution_data["arrivals"][str(k)] = {str(i): float(time) for i, time in arrivals.items()}

    # Save based on format type
    if format.lower() == "json":
        file_path = save_dir / f"{filename}.json"
        with open(file_path, "w") as f:
            json.dump(solution_data, f, indent=2)

    elif format.lower() == "pickle":
        file_path = save_dir / f"{filename}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(solution_data, f)

    else:
        raise ValueError(f"Unsupported format: {format}. Choose from 'json' or 'pickle'.")

    print(f"Solution saved to {file_path}")
    return str(file_path)


def load_solution(filepath):
    """
    Load a saved solution from a file.

    Args:
        filepath (str): Path to the saved solution file

    Returns:
        dict: The loaded solution data containing routes and arrivals
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Solution file not found: {filepath}")

    if filepath.suffix == ".json":
        with open(filepath, "r") as f:
            return json.load(f)

    elif filepath.suffix == ".pkl":
        with open(filepath, "rb") as f:
            return pickle.load(f)

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def apply_solution_to_model(model, solution_data):
    """
    Apply a loaded solution to a model instance for visualization or analysis.

    Args:
        model: A model instance to apply the solution to
        solution_data: Solution data loaded from a file

    Returns:
        The model instance with the solution applied
    """
    # Convert route data back to expected format
    model.routes = {}
    for k, routes in solution_data["routes"].items():
        # Convert string keys back to integers if they're numeric
        k_converted = int(k) if k.isdigit() else k

        # Convert string node IDs back to integers if they're numeric
        model.routes[k_converted] = []
        for i, j in routes:
            i_converted = int(i) if i.isdigit() else i
            j_converted = int(j) if j.isdigit() else j
            model.routes[k_converted].append((i_converted, j_converted))

    # Convert arrival times back to expected format
    model.arrivals = {}
    for k, arrivals in solution_data["arrivals"].items():
        # Convert string keys back to integers if they're numeric
        k_converted = int(k) if k.isdigit() else k

        model.arrivals[k_converted] = {}
        for i, time in arrivals.items():
            # Convert string node IDs back to integers if they're numeric
            i_converted = int(i) if i.isdigit() else i
            model.arrivals[k_converted][i_converted] = time

    print(f"Solution applied to model successfully")
    return model
