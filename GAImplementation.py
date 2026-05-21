"""Inner GA scheduler and JSON problem-loading helpers."""

import random
import json
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import islice
from deap import base, creator, tools
import numpy as np

from ga_config import load_config, resolve_config_path


DEFAULT_CONFIG = load_config()


def get_problem_settings(config):
    """Return the problem-building settings from the shared config."""
    return config["problem"]


def get_scheduler_settings(config):
    """Return the scheduler-specific settings from the shared config."""
    return config["scheduler"]


def get_scheduler_message_path_choices(scheduler_config, problem_config):
    """Return the available path-choice indices used by the inner GA genome."""
    configured_choices = scheduler_config.get("message_path_choices")
    if configured_choices:
        return list(configured_choices)

    return list(range(problem_config["k_shortest_paths"]))


def resolve_fallback_path_id(all_path_indexes_with_costs, scheduler_config):
    """Return a safe fallback path id even if the configured one is missing."""
    configured_fallback = str(scheduler_config["reconstruction"]["fallback_path_id"])
    if configured_fallback in all_path_indexes_with_costs:
        return configured_fallback

    return next(iter(all_path_indexes_with_costs))


def build_processor_index_map(processor_ids):
    """Map sparse processor ids from the JSON file onto dense internal indices."""
    return {processor_id: index for index, processor_id in enumerate(processor_ids)}


def validate_processor_assignment(task_allocation, processor_ids):
    """Fail fast if a task is assigned to a node that is not a real processor."""
    valid_processor_ids = set(processor_ids)
    invalid_processor_ids = sorted(
        {processor_id for processor_id in task_allocation if processor_id not in valid_processor_ids}
    )
    if invalid_processor_ids:
        raise ValueError(
            "Invalid processor ids in task allocation: "
            f"{invalid_processor_ids}. Expected only non-router processor ids from the platform."
        )


def Read_Parent_AM(json_data):     # Returning a dictionary about the Application graph
  AMx = json_data['application']
  return AMx

def Read_Parent_PM(json_data):     # Returning a dictionary about the Platform graph
  PMx = json_data['platform']
  return PMx


def construct_communication_costs_from_json(json_data):
  messages = json_data['application']['messages']
  communication_costs = {}

  for message in messages:
      sender = message['sender']
      receiver = message['receiver']
      size = message['size']

      if sender not in communication_costs:
          communication_costs[sender] = {receiver: size}
      else:
          communication_costs[sender][receiver] = size

  return communication_costs

def construct_task_dag_from_json(APP_MODEL): # where APP_MODELis an instance from the function Read_Parent_AM(json_data)
    # this function returns 2 lists one for task_dag (list of lists) for the successors
    # Another list tis the wcet_values showing the worst excution times for each job
    jobs = APP_MODEL['jobs']
    messages = APP_MODEL['messages']

    num_tasks = len(jobs)

    # Create a mapping of sender and receiver tasks for each message
    message_mapping = {}
    for message in messages:
        sender = message['sender']
        receiver = message['receiver']
        if sender not in message_mapping:
            message_mapping[sender] = [receiver]
        else:
            message_mapping[sender].append(receiver)

    # Create the task DAG
    task_dag = [[] for _ in range(num_tasks)]

    for job_id, successors in message_mapping.items():
        task_dag[job_id] = successors

    # Extract the WCET values
    wcet_values = [job['processing_times'] for job in jobs]

    return task_dag, wcet_values


def extract_message_list(APP_MODEL):
  # Returns a list of dictionaries for each meassage attribute where the keys for each message will be
  # id,sender,reciever,size
    messages = APP_MODEL['messages']                              # feteching messages (list of dictionaries)
    task_ids = [job['id'] for job in APP_MODEL['jobs']]           # creating a list of task_ids
    message_list = []                                                            # Initializing a list
    for msg in messages:
        sender_id = task_ids.index(msg['sender'])
        receiver_id = task_ids.index(msg['receiver'])
        message_size = msg['size']
        message_id = msg['id']
        message_info = {
            'id': message_id,
            'sender': sender_id,
            'receiver': receiver_id,
            'size': message_size
        }
        message_list.append(message_info)
    return message_list

def compute_makespan(schedule):    # passing the Re-construction function result
    # Extract end times from the schedule
    end_times = [info[2] for info in schedule.values()]
    # The makespan is the maximum end time
    makespan = max(end_times)
    return makespan

def plot_schedule(schedule):
    fig, ax = plt.subplots()

    processors = sorted(list(set([processor for processor, _, _ in schedule.values()])))
    colors = plt.cm.tab10(np.linspace(0, 1, len(processors)))

    for task, (processor, start_time, end_time) in schedule.items():
        color = colors[processor]
        ax.plot([start_time, end_time], [task, task], label=f'Processor {processor}', linewidth=10, marker='o', color=color)

    ax.set_xlabel('Time')
    ax.set_ylabel('Task')

    # Calculate the makespan and set x-axis limit
    makespan = max(end_time for _, (_, _, end_time) in schedule.items())
    ax.set_xlim(0, makespan)

    # Set y-axis ticks and labels
    plt.yticks(range(len(schedule)))
    plt.grid()
    plt.title("Task Schedule")

    # Create a custom legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()

def construct_graph_from_json(PLAT_MODEL):         # Function for constructing the PM , returns a graph object
    # Extract nodes and links from JSON data
    nodes = PLAT_MODEL['nodes']
    links = PLAT_MODEL['links']

    # Create an empty graph
    graph = nx.Graph()

    # Add nodes to the graph
    for node in nodes:
        node_id = node['id']
        node_type = 'processor' if not node['is_router'] else 'switch'
        graph.add_node(node_id, node_type=node_type)

    # Add edges (links) to the graph
    for link in links:
        start = link['start']
        end = link['end']
        graph.add_edge(start, end)

    return graph

def generate_all_path_indexes_with_costs(graph):
    processors = [node for node, data in graph.nodes(data=True) if data['node_type'] == 'processor']
    # creating a list with the processor nodes ids
    switches = [node for node, data in graph.nodes(data=True) if data['node_type'] == 'switch']
    # creating a list with the switche nodes ids

    def find_all_paths(source, target, path=[]): # A recursive function to find all paths between a give source node and target node, source and target are the iterables from the below for loop
      # This function uses a depth first search (DFS) to explore the paths
        path = path + [source]                    # Appending the "source" node to the path list, to keep track of nodes vistied in the list
        if source == target:
            return [path]
        if source not in graph:                    # checks for node presence in graph
            return []
        paths = []
        for node in graph[source]:                 # iterating through the current neighbors in graph
            if node not in path:
                newpaths = find_all_paths(node, target, path)  # calling the fuction recursively taking the neighbor node as a source node , target node is the same as the old one, path= path calculated so far
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    path_indexes = {}                             # Initialzing a dict. to store results (paths and costs)
    path_id = 0
    for source in processors:                     # iterating through the processors list in the PM [0,1,2]
        for target in processors:                 # iterating through processors list in PM [0,1,2], done to consider all pairs between source and target nodes
            if source == target:
                # Handle self-loop
                path_indexes[path_id] = {"path": [source, source], "cost": 0}
                path_id += 1
            else:
                all_paths = find_all_paths(source, target)
                all_paths = [path for path in all_paths if any(node in path for node in switches)] # filtering the paths to keep the ones with only one switch node
                if all_paths:
                    for path in all_paths:
                        # Compute the cost as the number of edges in the path
                        path_cost = len(path) - 1                                   # Computing the cost by subtracting 1 from the number of nodes in the path.
                        # Add the path, its ID, and its cost to the result
                        path_indexes[path_id] = {"path": path, "cost": path_cost}
                        path_id += 1
    return path_indexes



def get_processor_ids(data):
    processor_ids = [node['id'] for node in data['platform']['nodes'] if not node['is_router']]
    return processor_ids


def plot_schedule_w_dep(schedule, message_list):
    fig, ax = plt.subplots(figsize=(15, 10))  # Increase figure size

    processors = sorted(list(set([processor for processor, _, _ ,_ in schedule.values()])))
    colors = plt.cm.tab10(np.linspace(0, 1, len(processors)))

    for task, (processor, start_time, end_time,_) in schedule.items():
        color = colors[processors.index(processor)]
        ax.plot([start_time, end_time], [task, task], label=f'Processor {processor}', linewidth=10, marker='o', color=color)

    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Task', fontsize=14)

    # Calculate the makespan and set x-axis limit
    makespan = max(end_time for _, (_, _, end_time,_) in schedule.items())
    ax.set_xlim(0, makespan)

    # Set y-axis ticks and labels
    plt.yticks(range(len(schedule)), fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Task Schedule", fontsize=16)

    # Create a custom legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12)

    # Add arrows to represent messages
    for message in message_list:
        sender = message['sender']
        receiver = message['receiver']
        if sender in schedule and receiver in schedule:
            sender_end = schedule[sender][2]
            receiver_start = schedule[receiver][1]
            ax.annotate("",
                        xy=(receiver_start, receiver),
                        xytext=(sender_end, sender),
                        arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

    plt.tight_layout()
    plt.show()

# Define a function to find the k-shortest paths (using shortest path method)
def k_shortest_paths(G, source, target, k):
    return list(islice(nx.shortest_simple_paths(G, source, target), k))

# A function to calculate path cost based on number of routers in the list
def path_cost(G, path):
    return sum(1 for i in range(1, len(path)) if G.nodes[path[i]]['is_router'])

# Calculate the k-shortest paths with diversity between two processors
def diverse_k_shortest_paths(G, source, target, k):
    paths = k_shortest_paths(G, source, target, k)
    paths_with_costs = [(path, path_cost(G, path)) for path in paths]
    return paths_with_costs

def load_problem(input_json_path=None, config=None):
    """Load one example JSON file and build the data structures used by the GA."""
    config = config or load_config()
    if input_json_path is None:
        input_json_path = resolve_config_path(
            config,
            config["paths"]["default_input_json"],
        )
    else:
        input_json_path = resolve_config_path(config, input_json_path)

    with input_json_path.open("r", encoding="utf-8") as f:
        loaded_json_data = json.load(f)

    loaded_AMx = Read_Parent_AM(loaded_json_data)
    loaded_PMx = Read_Parent_PM(loaded_json_data)
    loaded_successors, loaded_processing_times = construct_task_dag_from_json(loaded_AMx)
    loaded_message_list = extract_message_list(loaded_AMx)
    loaded_platform_graph_object = construct_graph_from_json(loaded_PMx)
    loaded_processor_ids = get_processor_ids(loaded_json_data)

    # Extract nodes and links from the JSON data.
    loaded_nodes = loaded_json_data["platform"]["nodes"]
    loaded_links = loaded_json_data["platform"]["links"]

    loaded_graph = nx.Graph()
    for node in loaded_nodes:
        loaded_graph.add_node(node["id"], is_router=node["is_router"])

    for link in loaded_links:
        loaded_graph.add_edge(link["start"], link["end"])

    loaded_processor_nodes = [node["id"] for node in loaded_nodes if not node["is_router"]]

    # Dictionary to store the paths between processors with separate path_id and sub_path_id.
    loaded_paths_dict = {}

    # Dictionary to store the merged paths with merged_id.
    loaded_merged_paths_dict = {}

    problem_settings = get_problem_settings(config)
    k = problem_settings["k_shortest_paths"]
    self_loop_cost = problem_settings["self_loop_cost"]
    path_id = 1

    for i in range(len(loaded_processor_nodes)):
        for j in range(i + 1, len(loaded_processor_nodes)):
            source = loaded_processor_nodes[i]
            target = loaded_processor_nodes[j]
            paths = diverse_k_shortest_paths(loaded_graph, source, target, k)

            # Create a sub-dictionary for this path_id.
            sub_paths_dict = {}
            for sub_path_id, (path, cost) in enumerate(paths):
                sub_paths_dict[sub_path_id] = {"path": path, "cost": cost}
                # Merge path_id and sub_path_id into a single merged_id as a string.
                merged_id = f"{path_id}{sub_path_id}"
                loaded_merged_paths_dict[merged_id] = {"path": path, "cost": cost}

            # Add the sub-dictionary to the main dictionary with path_id as key.
            loaded_paths_dict[path_id] = sub_paths_dict

            # Increment the path_id for the next pair of processors.
            path_id += 1

    # Adding self-loops after calculating the diverse paths.
    for i in range(len(loaded_processor_nodes)):
        source = loaded_processor_nodes[i]
        target = source  # Self-loop

        # Self-loop represented as [source, target] with a configurable cost.
        sub_paths_dict = {0: {"path": [source, target], "cost": self_loop_cost}}
        loaded_paths_dict[path_id] = sub_paths_dict

        # Merge path_id and sub_path_id into a single merged_id as a string for self-loop.
        merged_id = f"{path_id}0"
        loaded_merged_paths_dict[merged_id] = {
            "path": [source, target],
            "cost": self_loop_cost,
        }

        # Increment the path_id for the next processor.
        path_id += 1

    return {
        "INPUT_JSON_PATH": input_json_path,
        "json_data": loaded_json_data,
        "AMx": loaded_AMx,
        "PMx": loaded_PMx,
        "successors": loaded_successors,
        "processing_times": loaded_processing_times,
        "message_list": loaded_message_list,
        "PLATFORM_GRAPH_OBJECT": loaded_platform_graph_object,
        "n_tasks": len(loaded_processing_times),
        "processor_ids": loaded_processor_ids,
        "nodes": loaded_nodes,
        "links": loaded_links,
        "G": loaded_graph,
        "processor_nodes": loaded_processor_nodes,
        "paths_dict": loaded_paths_dict,
        "merged_paths_dict": loaded_merged_paths_dict,
    }


def reconstruct_schedule_with_precedenceX_updated(
    processor_ids,
    task_allocation,
    node_list,
    processing_times,
    message_list,
    message_path_index,
    all_path_indexes_with_costs,
    message_priority_ordering,
    scheduler_config=None,
):
    # Create a deep copy of the message_list to avoid modifying the original data
    message_list_copy = deepcopy(message_list)

    scheduler_config = scheduler_config or get_scheduler_settings(DEFAULT_CONFIG)
    fallback_path_id = resolve_fallback_path_id(
        all_path_indexes_with_costs,
        scheduler_config,
    )
    validate_processor_assignment(task_allocation, processor_ids)
    processor_index_map = build_processor_index_map(processor_ids)
    processor_slot_count = len(processor_ids)

    schedule = {}  # Dictionary to store the new schedule
    task_completion_times = [0] * len(node_list)  # List to store task completion times
    message_dict = defaultdict(list)  # Dictionary to store messages by receiver

    # Create a dictionary to map message ID to priority
    message_priority_dict = {message_id: priority for priority, message_id in enumerate(message_priority_ordering)}

    # Create a mapping from task ID to processor
    task_to_processor = {task_id: processor for task_id, processor in zip(node_list, task_allocation)}

    # Replace sender and receiver in message list with the corresponding processor
    updated_message_list = []  # Container for mapping the messages between the processors
    for message in message_list_copy:
        updated_message = {
            'id': message['id'],
            'sender': task_to_processor[message['sender']],
            'receiver': task_to_processor[message['receiver']],
            'size': message['size']
        }
        updated_message_list.append(updated_message)
    
    # Initialize the results list
    message_to_path_mapping = []

    # Initialize a dictionary to track the usage of path_ids
    path_usage = {path_id: 0 for path_id in message_path_index}

    # Loop through each message and find the corresponding path
    for i, message in enumerate(updated_message_list):
        sender = message['sender']
        receiver = message['receiver']
        # Iterate through the message_path_index to find a matching path
        for path_id in message_path_index:
            path_info = all_path_indexes_with_costs[path_id]
            path = path_info['path']
            
            # Check if the sender and receiver match the path's start and end
            if (path[0] == sender and path[-1] == receiver) or (path[-1] == sender and path[0] == receiver):
                # Check if this path has been used as many times as it appears in message_path_index
                if path_usage[path_id] < message_path_index.count(path_id):
                    message_to_path_mapping.append({'message_id': message['id'], 'path_id': path_id})
                    path_usage[path_id] += 1
                    break
    
    for idx, message in enumerate(message_list_copy):
        # Find the corresponding path_id from message_to_path_mapping
        message_mapping = next((m for m in message_to_path_mapping if m['message_id'] == message['id']), None)
        
        if message_mapping:
            path_id = message_mapping['path_id']
        else:
            path_id = fallback_path_id

        # Decode the path and cost using the path_id
        path = all_path_indexes_with_costs[path_id]["path"]
        path_cost = all_path_indexes_with_costs[path_id]["cost"]

        # Adjust the size of the message with the cost of the path
        message["size"] += path_cost

        # Append a tuple containing the sender, message size, message priority, path_id, and message_id to the receiver's list in the message_dict
        message_dict[message["receiver"]].append((message["sender"], message["size"], message_priority_dict[message["id"]], path_id, message["id"]))

        
    # Sort each receiver's list in message_dict by message priority
    for receiver, messages in message_dict.items():
        message_dict[receiver] = sorted(messages, key=lambda x: x[2])

    current_time_per_processor = [0] * processor_slot_count  # Dense processor availability timeline
    
    completed_tasks = set()  # Set of completed tasks
    ready_tasks = set(range(len(node_list)))  # Set of tasks ready to be processed

    while ready_tasks:
        task = ready_tasks.pop()
        task_id = node_list[task]
        processor = task_allocation[task]
        processor_slot = processor_index_map[processor]

        predecessors = message_dict[task_id]

        if all(p in completed_tasks for p, _, _, _, _ in predecessors):
            if predecessors:
                # Calculate the latest time when all predecessor tasks are completed including message sizes
                latest_predecessor_completion = max(
                    task_completion_times[sender] + size for sender, size, _, _, _ in predecessors
                )
                # Compare it with the current processor time and take the maximum
                start_time = max(
                    current_time_per_processor[processor_slot],
                    latest_predecessor_completion,
                )
            else:
                # If there are no predecessors, start at the current processor time
                start_time = current_time_per_processor[processor_slot]

            # Calculate the total size of all messages from predecessors
            total_message_size = sum(size for _, size, _, _, _ in predecessors)

            # Calculate the end_time considering processing time and total message sizes
            end_time = start_time + processing_times[task_id] + total_message_size

            # Record the path information used by the predecessors
            path_info = [(sender, path_id, message_id) for sender, _, _, path_id, message_id in predecessors]
            schedule[task_id] = (processor, start_time, end_time, path_info)
            task_completion_times[task_id] = end_time

            # Update the current time of the processor to the end time of this task
            current_time_per_processor[processor_slot] = end_time
            completed_tasks.add(task_id)
        else:
            ready_tasks.add(task)
    return schedule


def NEW_GA_V2(
    processor_ids,
    processing_times,
    message_list,
    all_path_indexes_with_costs,
    pop_size,
    cxpb,
    mutpb,
    ngen,
    random_seed=None,
    scheduler_config=None,
    return_generation_history=False,
):
    if random_seed is not None:
        random.seed(random_seed)

    scheduler_config = scheduler_config or get_scheduler_settings(DEFAULT_CONFIG)
    problem_settings = get_problem_settings(DEFAULT_CONFIG)
    message_path_choices = get_scheduler_message_path_choices(
        scheduler_config,
        problem_settings,
    )
    selection_tournament_size = scheduler_config["selection"]["tournament_size"]
    task_order_mutation_probability = scheduler_config["mutation"][
        "task_order_probability"
    ]
    processor_allocation_mutation_probability = scheduler_config["mutation"][
        "processor_allocation_probability"
    ]
    message_priority_shuffle_probability = scheduler_config["mutation"][
        "message_priority_shuffle_probability"
    ]
    message_path_index_mutation_probability = scheduler_config["mutation"][
        "message_path_index_probability"
    ]

    # Create a FitnessMin class for minimization and set the weights to (-1.0,).
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-100.0,))
    # Create the Individual class as a list with the FitnessMin class as the fitness attribute.
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()

    # Parameters
    num_tasks = len(processing_times)  # Number of jobs in task graph
    num_message = len(message_list)    # Number of messages
    predefined_processors = processor_ids    # Predefined processor values
    message_list_ids = [message['id'] for message in message_list]

    def split_individual(individual):
        """Split a flat DEAP individual into the four scheduler genome segments."""
        task_order = individual[:num_tasks]
        processor_allocation = individual[num_tasks:num_tasks + num_tasks]
        message_priority_ordering = individual[
            num_tasks + num_tasks:num_tasks + num_tasks + num_message
        ]
        message_path_index = individual[num_tasks + num_tasks + num_message:]
        return task_order, processor_allocation, message_priority_ordering, message_path_index

    def validate_task_order(task_order, context):
        """Fail fast if the permutation-encoded task order becomes invalid."""
        duplicate_task_ids = sorted(
            task_id for task_id, count in Counter(task_order).items() if count > 1
        )
        missing_task_ids = sorted(set(range(num_tasks)) - set(task_order))
        if len(task_order) != num_tasks or duplicate_task_ids or missing_task_ids:
            raise ValueError(
                "Invalid task-order genome during "
                f"{context}: duplicates={duplicate_task_ids}, missing={missing_task_ids}"
            )

    # Define the initialization function for the task order in the individual
    def init_task_order():
        return random.sample(range(num_tasks), num_tasks)  # Returns a list of random numbers in the range of num_tasks

    # Define the processor allocation initialization function
    def processor_allocation(n_task, predefined_values):
        return [random.choice(predefined_values) for _ in range(n_task)]  # Random list for processor allocation from predefined values


    def message_priority_ordering(n_messages, defined_values):
        # Ensure n_messages is positive and not greater than the number of defined values
        if n_messages <= 0 or n_messages > len(defined_values):
            raise ValueError("n_messages must be greater than 0 and not exceed the number of defined values")
        # Use random.sample to ensure unique values
        return random.sample(defined_values, n_messages)
    

    # Add the message path index initialization function
    def init_message_path_index(n_messages):
        return [random.choice(message_path_choices) for _ in range(n_messages)]

    # Combined Individual with new message_path_index
    def create_individual():
        individual = []
        individual.extend(toolbox.task_order())
        individual.extend(toolbox.processor_allocation())
        individual.extend(toolbox.message_priority_ordering())
        individual.extend(toolbox.message_path_index())  # Include message_path_index
        validate_task_order(individual[:num_tasks], "individual initialization")
        return individual

    # Register the initialization functions in the DEAP toolbox
    toolbox.register("task_order", init_task_order)
    toolbox.register("processor_allocation", processor_allocation, n_task=num_tasks, predefined_values=predefined_processors)
    toolbox.register("message_priority_ordering", message_priority_ordering, n_messages=num_message,defined_values = message_list_ids )
    toolbox.register("message_path_index", init_message_path_index, n_messages=num_message)
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function
    def compute_makespan(schedule):
        # Extract end times from the schedule
        end_times = [info[2] for info in schedule.values()]
        # The makespan is the maximum end time
        makespan = max(end_times)
        return makespan
    
    def ComputeMappingsAndPaths(message_list, tasks, processors, message_orderings, path_indices):
        message_list_copy = deepcopy(message_list)
        
        # Create a dictionary that maps task ids to processor ids
        task_to_processor = {task: processors[i] for i, task in enumerate(tasks)}
        
        updated_message_list = []  # Container for mapping the messages between the processors
        
        # Create a mapping between message IDs (orderings) and path indices
        message_to_path_mapping = {message_orderings[i]: path_indices[i] for i in range(len(message_orderings))}
        
        for message in message_list_copy:
            # Find the corresponding path index using the message ID
            path_index = message_to_path_mapping.get(message['id'], None)

            if task_to_processor[message['sender']] == task_to_processor[message['receiver']]:
                path_index = message_path_choices[0]
            
            updated_message = {
                'id': message['id'],
                'sender': task_to_processor[message['sender']],
                'receiver': task_to_processor[message['receiver']],
                'size': message['size'],
                'path_index': path_index  # Add the path index to the updated message
            }
            updated_message_list.append(updated_message)
        
        return updated_message_list


    def find_suitable_paths(updated_message_list, merged_paths_dict):
        selected_paths = []
        
        for message in updated_message_list:
            sender = message['sender']
            receiver = message['receiver']
            path_index_str = str(message['path_index'])

            valid_path_ids = set()
            
            # Iterate through merged_paths_dict to find all valid paths
            for pid, details in merged_paths_dict.items():
                path = details['path']
                
                if ((path[0] == sender and path[-1] == receiver) or 
                    (path[0] == receiver and path[-1] == sender)) and pid.endswith(path_index_str):
                    valid_path_ids.add(pid)
            
            # Select the valid path_id if found, else set to None
            path_id = next(iter(valid_path_ids), None)
            
            selected_paths.append(path_id)
        
        return selected_paths

    def evaluate(individual, processing_times, message_list, all_path_indexes_with_costs):
        task_order, processor_allocation, message_priority_ordering, message_path_index = (
            split_individual(individual)
        )
        validate_task_order(task_order, "evaluation")
        updated_list  = ComputeMappingsAndPaths(message_list, task_order, processor_allocation,message_priority_ordering,message_path_index)
        Selected_paths = find_suitable_paths(updated_list, all_path_indexes_with_costs)


        schedule = reconstruct_schedule_with_precedenceX_updated(
            processor_ids,
            processor_allocation,
            task_order,
            processing_times,
            message_list,
            Selected_paths,
            all_path_indexes_with_costs,
            message_priority_ordering,
            scheduler_config=scheduler_config,
        )
        makespan = compute_makespan(schedule)
        fitness =  makespan  # Fitness is the inverse of makespan for minimization
        return fitness,

    def build_generation_history_row(population, generation):
        generation_fitnesses = [ind.fitness.values[0] for ind in population]
        generation_best = tools.selBest(population, 1)[0]
        return {
            "generation": generation,
            "generation_best_makespan": generation_best.fitness.values[0],
            "generation_avg_makespan": sum(generation_fitnesses) / len(generation_fitnesses),
            "generation_worst_makespan": max(generation_fitnesses),
        }

    # Register the evaluation function
    toolbox.register("evaluate", evaluate, processing_times=processing_times, message_list=message_list, all_path_indexes_with_costs=all_path_indexes_with_costs)

    # Register selection operator
    toolbox.register("select", tools.selTournament, tournsize=selection_tournament_size)

    # Register the crossover operator (Permutation encoded)
    def mate(ind1, ind2, task_order_len, processor_allocation_len, message_priority_ordering_len):
        # Recombine each genome segment in place so crossover probability has a real effect.
        task_order_1 = ind1[:task_order_len]
        task_order_2 = ind2[:task_order_len]
        tools.cxOrdered(task_order_1, task_order_2)
        validate_task_order(task_order_1, "crossover child 1")
        validate_task_order(task_order_2, "crossover child 2")
        ind1[:task_order_len] = task_order_1
        ind2[:task_order_len] = task_order_2

        processor_start = task_order_len
        processor_end = processor_start + processor_allocation_len
        processor_allocation_1 = ind1[processor_start:processor_end]
        processor_allocation_2 = ind2[processor_start:processor_end]
        tools.cxUniform(processor_allocation_1, processor_allocation_2, indpb=0.5)
        ind1[processor_start:processor_end] = processor_allocation_1
        ind2[processor_start:processor_end] = processor_allocation_2

        message_priority_start = processor_end
        message_priority_end = message_priority_start + message_priority_ordering_len
        message_priority_ordering_1 = ind1[message_priority_start:message_priority_end]
        message_priority_ordering_2 = ind2[message_priority_start:message_priority_end]
        tools.cxOrdered(message_priority_ordering_1, message_priority_ordering_2)
        ind1[message_priority_start:message_priority_end] = message_priority_ordering_1
        ind2[message_priority_start:message_priority_end] = message_priority_ordering_2

        message_path_index_1 = ind1[message_priority_end:]
        message_path_index_2 = ind2[message_priority_end:]
        tools.cxUniform(message_path_index_1, message_path_index_2, indpb=0.5)
        ind1[message_priority_end:] = message_path_index_1
        ind2[message_priority_end:] = message_path_index_2

        return ind1, ind2

    toolbox.register(
        "mate",
        mate,
        task_order_len=num_tasks,
        processor_allocation_len=num_tasks,
        message_priority_ordering_len=num_message,
    )

    # Mutation functions
    def mutation_task_order(individual, task_order_len):
        task_order = deepcopy(individual[:task_order_len])
        if task_order_len > 1 and random.random() < task_order_mutation_probability:
            indices = random.sample(range(task_order_len), 2)  # Select two random indices
            task_order[indices[0]], task_order[indices[1]] = task_order[indices[1]], task_order[indices[0]]  # Swap the elements at the selected indices
        validate_task_order(task_order, "task-order mutation")
        individual[:task_order_len] = task_order
        return individual,

    def mutation_processor_allocation(individual, task_order_len, processor_allocation_len, predefined_values):
        processor_allocation = deepcopy(individual[task_order_len:task_order_len + processor_allocation_len])
        for i in range(len(processor_allocation)):
            if random.random() < processor_allocation_mutation_probability:
                processor_allocation[i] = random.choice(predefined_values)
        individual[task_order_len:task_order_len + processor_allocation_len] = processor_allocation
        return individual,


    def mutation_message_priority_ordering(individual, task_order_len, processor_allocation_len, message_path_index_len, message_list):
        # Extract the message priority ordering part of the individual
        message_priority_ordering = deepcopy(individual[task_order_len + processor_allocation_len : task_order_len + processor_allocation_len + message_path_index_len])
        
        # Shuffle the message priority ordering
        tools.mutShuffleIndexes(message_priority_ordering, indpb=message_priority_shuffle_probability)
        
        # Ensure uniqueness
        unique_ids = list({msg['id'] for msg in message_list})  # Get unique IDs from message_list
        random.shuffle(unique_ids)  # Shuffle to introduce randomness

        # Replace duplicates in the shuffled message_priority_ordering
        seen = set()
        for i in range(len(message_priority_ordering)):
            if message_priority_ordering[i] in seen:
                # If the current element is a duplicate, replace it with an unused unique ID
                for uid in unique_ids:
                    if uid not in seen:
                        message_priority_ordering[i] = uid
                        seen.add(uid)
                        break
            else:
                seen.add(message_priority_ordering[i])

        # Ensure the list is exactly of length message_path_index_len
        message_priority_ordering = message_priority_ordering[:message_path_index_len]

        # Replace the individual's message priority ordering with the updated unique list
        individual[task_order_len + processor_allocation_len: task_order_len + processor_allocation_len + message_path_index_len] = message_priority_ordering
        
        return individual,


    def mutation_message_path_index(individual, task_order_len, processor_allocation_len, message_priority_ordering_len, message_path_index_len):
        message_path_index = deepcopy(individual[task_order_len + processor_allocation_len + message_priority_ordering_len:])
        for i in range(len(message_path_index)):
            if random.random() < message_path_index_mutation_probability:
                message_path_index[i] = random.choice(message_path_choices)
        individual[task_order_len + processor_allocation_len + message_priority_ordering_len:] = message_path_index
        return individual,


    # Register the mutation operators
    toolbox.register("mutate_task_order", mutation_task_order, task_order_len=num_tasks)
    toolbox.register("mutate_processor_allocation", mutation_processor_allocation, task_order_len=num_tasks, processor_allocation_len=num_tasks, predefined_values=predefined_processors)
    toolbox.register("mutate_message_priority_ordering", mutation_message_priority_ordering, task_order_len=num_tasks, processor_allocation_len=num_tasks, message_path_index_len=num_message,message_list =message_list)
    toolbox.register("mutate_message_path_index", mutation_message_path_index, task_order_len=num_tasks, processor_allocation_len=num_tasks,message_priority_ordering_len=num_message, message_path_index_len=num_message)


    # Create an initial population
    pop = toolbox.population(n=pop_size)

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Crossover probability and mutation probability
    CXPB, MUTPB = cxpb, mutpb

    # Extract all the fitnesses
    fits = [ind.fitness.values[0] for ind in pop]
    generation_history = [build_generation_history_row(pop, generation=0)]

    NGEN = ngen # Number of generations
    # Begin the evolution
    for g in range(NGEN):
      
        # A new generation
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                child1, child2 = toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate_task_order(mutant)
                toolbox.mutate_processor_allocation(mutant)
                toolbox.mutate_message_priority_ordering(mutant)
                toolbox.mutate_message_path_index(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        generation_history.append(build_generation_history_row(pop, generation=g + 1))

    # Print the best individual
    best_ind = tools.selBest(pop, 1)[0]

    task_order_, processor_allocation_, message_priority_ordering_, message_path_index_ = (
        split_individual(best_ind)
    )
    updated_list_ = ComputeMappingsAndPaths(message_list, task_order_, processor_allocation_,message_priority_ordering_,message_path_index_)
    Selected_paths_ = find_suitable_paths(updated_list_, all_path_indexes_with_costs)


    scheduleFinal = reconstruct_schedule_with_precedenceX_updated(
        processor_ids,
        processor_allocation_,
        task_order_,
        processing_times,
        message_list,
        Selected_paths_,
        all_path_indexes_with_costs,
        message_priority_ordering_,
        scheduler_config=scheduler_config,
    )
    
    Final_genome = [task_order_, processor_allocation_,  message_priority_ordering_,message_path_index_]

    makespan_final = compute_makespan(scheduleFinal)
    if return_generation_history:
        return makespan_final, scheduleFinal, Final_genome, generation_history
    return makespan_final, scheduleFinal, Final_genome


