from random import uniform, randint, choice, random, gauss
import pickle as pkl
import math

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.

    Computes the element-wise maximum of 0 and the input value.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        ReLU output value (max(0, x)).
    """
    return max(0.0, x)

def sigmoid(x):
    """
    Sigmoid (logistic) activation function.

    Computes the sigmoid function: 1 / (1 + exp(-x)).

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Sigmoid output value, ranging between 0 and 1.
    """
    # Clamp input to avoid overflow in math.exp() for very large negative numbers
    # and underflow for very large positive numbers.
    # This helps maintain numerical stability.
    clamped_x = max(-700.0, min(700.0, x)) # Values outside this range are effectively 0 or 1
    return 1 / (1 + math.exp(-clamped_x))

def select_activation(name):
    """
    Selects and returns an activation function based on its name.

    Parameters
    ----------
    name : str
        Name of the activation function (e.g., 'relu', 'sigmoid').

    Returns
    -------
    function
        The corresponding activation function.

    Raises
    ------
    ValueError
        If the provided name does not match any known activation function.
    """
    if name == 'relu':
        return relu
    elif name == 'sigmoid':
        return sigmoid
    else:
        raise ValueError(f'Unknown activation function: {name}')

class NEATConfig:
    """
    Configuration class for the algorithm.

    Stores various hyperparameters and settings that control the evolution process,
    genome structure, mutation rates and speciation.

    Attributes
    ----------
    population_size : int
        The number of genomes in the population.
    genome_shape : tuple[int, int]
        A tuple representing (number_of_input_nodes, number_of_output_nodes)
        for initial genome creation.
    hid_node_activation : str
        The name of the activation function used for hidden nodes (e.g., 'relu').
    out_node_activation : str
        The name of the activation function used for output nodes (e.g., 'sigmoid').
    max_weight : float
        The maximum weight for the hidden nodes.
    min_weight : float
        The minimum weight for the hidden nodes.
    add_node_mutation_prob : float
        The probability of adding a new node during mutation.
    add_conn_mutation_prob : float
        The probability of adding a new connection during mutation.
    num_elites : int
        The number of best genomes from each species to be carried over directly
        to the next generation without mutation (elitism).
    selection_share : float
        The fraction of top genomes within a species eligible for reproduction (crossover).
    sigma : float
        The standard deviation for the Gaussian noise added during weight perturbation mutation.
    perturb_prob : float
        The probability that an existing connection's weight will be perturbed.
    reset_prob : float
        The probability that an existing connection's weight will be completely reset
        (within the perturb_prob chance).
    species_threshold : float
        The compatibility distance threshold used for speciation. Genomes with
        a distance below this threshold are considered to be in the same species.
    c1 : float
        Coefficient for excess genes in the compatibility distance calculation.
    c2 : float
        Coefficient for disjoint genes in the compatibility distance calculation.
    c3 : float
        Coefficient for average weight difference of matching genes in the
        compatibility distance calculation.
    save_path : str
        The directory path where results (like the best genome) should be saved.
    """
    def __init__(
            self,
            population_size=100,
            genome_shape=(1,1),
            hid_node_activation='relu',
            out_node_activation='sigmoid',
            max_weight=1.0,
            min_weight=-1.0,
            add_node_mutation_prob=0.05,
            add_conn_mutation_prob=0.08,
            num_elites=1,
            selection_share=0.2,
            sigma=0.1,
            perturb_prob=0.8,
            reset_prob=0.1,
            species_threshold=3.0,
            c1=1.0,
            c2=1.0,
            c3=0.4,
            save_path="./",
    ):
        """
        Initializes the NEATConfig object with specified or default parameters.
        """
        self.genome_shape = genome_shape
        self.add_node_mutation_prob = add_node_mutation_prob
        self.add_conn_mutation_prob = add_conn_mutation_prob
        self.sigma = sigma
        self.perturb_prob = perturb_prob
        self.reset_prob = reset_prob
        self.hid_node_activation = hid_node_activation
        self.out_node_activation = out_node_activation
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.species_threshold = species_threshold
        self.population_size = population_size
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_elites = num_elites
        self.selection_share = selection_share
        self.save_path = save_path

    def __repr__(self):
        """
        Returns a string representation of the configuration settings.

        Returns
        -------
        str
            A multi-line string listing all configuration parameters and their values.
        """
        str_repr = ""
        for key, value in self.__dict__.items():
            str_repr += f"{key}: {value}\n"
        return str_repr

class Connection:
    """
    Represents a connection gene in a genome.

    A connection links two nodes (an input node and an output node) and has
    an associated weight, an enabled status, and a unique innovation ID.

    Attributes
    ----------
    in_node : Node
        The node from which this connection originates.
    out_node : Node
        The node to which this connection leads.
    id : int
        The historical innovation number (marker) for this connection gene.
        Unique across the entire evolution history for a specific (in_node, out_node) pair.
    weight : float
        The strength of the connection.
    enabled : bool
        Indicates whether the connection is active in the network. Disabled
        connections are ignored during forward propagation but are kept for
        potential re-enablement or historical tracking.
    """
    def __init__(self, in_node, out_node, id=0):
        """
        Initializes a new connection gene.

        Parameters
        ----------
        in_node : Node
            The starting node for the connection.
        out_node : Node
            The ending node for the connection.
        id : int, optional
            The innovation ID for this connection. Defaults to 0, but should typically
            be assigned by the Population's tracking mechanism.
        """
        self.in_node = in_node
        self.out_node = out_node
        self.id = id
        self.weight = uniform(-1, 1) # Initialize weight randomly between -1 and 1
        self.enabled = True # Connections start enabled by default

    def copy(self):
        """
        Creates and returns a deep copy of this connection.

        The new connection shares the same attributes but is a distinct object.
        Node references within the copied connection initially point to the original
        nodes and need to be updated if copied within a genome context.

        Returns
        -------
        Connection
            A new Connection object with identical attributes.
        """
        conn = Connection(self.in_node, self.out_node, self.id)
        conn.enabled = self.enabled
        conn.weight = self.weight
        return conn

    def __repr__(self):
        """
        Returns a string representation of the connection.

        Returns
        -------
        str
            String detailing the connection's ID, connected node IDs, weight, and enabled status.
        """
        return f"id: {self.id}: nodes: {self.in_node.id} -> {self.out_node.id} (W:{self.weight:.2f} E:{self.enabled})"

class Node:
    """
    Represents a node gene in a genome.

    Nodes can be of type 'input', 'output', or 'hidden'. Each node has a unique ID
    and potentially an activation function. Its value is computed during the
    network's forward pass.

    Attributes
    ----------
    type : str
        The type of the node ('input', 'hidden', 'output').
    id : int
        A unique identifier for this node across the population.
    activation : str or None
        The name of the activation function associated with this node (e.g., 'relu', 'sigmoid').
        Input nodes typically have no activation function (None).
    value : float
        The computed output value of the node after activation during the forward pass.
        Initialized to 0.0.
    """
    def __init__(self, type, id, activation=None):
        """
        Creates a new node gene.

        Parameters
        ----------
        type : str
            The type of the node ('input', 'hidden', 'output').
        id : int
            The unique ID for the node.
        activation : str, optional
            The name of the activation function. Defaults to None.
        """
        self.type = type
        self.value = 0.0 # Runtime value, reset before each forward pass
        self.activation = activation
        self.id = id

    def copy(self):
        """
        Creates and returns a deep copy of this node.

        The new node shares the same attributes (except value, which is reset)
        but is a distinct object.

        Returns
        -------
        Node
            A new Node object with identical attributes (type, id, activation).
        """
        node = Node(self.type, self.id, self.activation)
        # Value is stateful and shouldn't be copied directly, reset in new node
        return node

    def __hash__(self):
        """
        Computes the hash based on the node's unique ID.

        Allows nodes to be used in sets or as dictionary keys.
        """
        return hash(self.id)

    def __eq__(self, other):
        """
        Checks equality based on the node's unique ID.

        Two nodes are considered equal if they have the same ID.
        """
        return isinstance(other, Node) and self.id == other.id

    def __repr__(self):
        """
        Returns a string representation of the node.

        Returns
        -------
        str
            String detailing the node's ID, type, and activation function.
        """
        return f"Node(id={self.id}, type={self.type}, act={self.activation})"

class Genome:
    """
    Represents a single individual (a neural network) in the population.

    A genome consists of a set of node genes and connection genes. It defines
    the structure and parameters of the neural network.

    Attributes
    ----------
    connections : dict[int, Connection]
        A dictionary mapping connection innovation IDs to Connection objects.
    nodes : dict[int, Node]
        A dictionary mapping node IDs to Node objects.
    population : Population
        A reference to the Population object this genome belongs to. Used for
        accessing global innovation numbers and configuration.
    fitness : float
        The evaluated fitness score of the genome. Higher is typically better.
    shape : tuple[int, int]
        The (input_size, output_size) of the network defined by this genome.
    config : NEATConfig or None
        A reference to the configuration object, providing hyperparameters.
    """
    def __init__(self, population, shape=(1,1), connections=None, nodes=None, config=None):
        """
        Creates a new genome.

        Can be initialized either with a basic shape (for creating a minimal starting genome)
        or with existing sets of connections and nodes (e.g., for crossover or copying).

        Parameters
        ----------
        population : Population
            The population this genome belongs to.
        shape : tuple[int, int], optional
            The (input_size, output_size) shape used for default initialization.
            Defaults to (1, 1). Only used if `connections` and `nodes` are None.
            If `config` is provided, `config.genome_shape` overrides this.
        connections : dict[int, Connection], optional
            An existing dictionary of connections. If provided along with `nodes`,
            the `shape` parameter is ignored. Defaults to None.
        nodes : dict[int, Node], optional
            An existing dictionary of nodes. If provided along with `connections`,
            the `shape` parameter is ignored. Defaults to None.
        config : NEATConfig, optional
            The configuration object. If provided, its settings (like genome_shape,
            activation functions) are used. Defaults to None.

        Raises
        ------
        AssertionError
            If initialization parameters are inconsistent (e.g., providing neither
            shape nor connections/nodes).
        """
        # Determine if using default shape initialization or providing components
        using_config_shape = config is not None and config.genome_shape != (1,1)
        using_param_shape = shape != (1,1) and not using_config_shape
        default_init = (using_config_shape or using_param_shape) and connections is None and nodes is None

        using_config_default = config is not None and config.genome_shape == (1,1)
        using_param_default = shape == (1,1) and not using_config_default
        conn_node_init = (using_config_default or using_param_default) and connections is not None and nodes is not None

        # Ensure exactly one initialization method is indicated
        assert default_init or conn_node_init, \
            "Provide either shape (or config with shape) for default init, or connections and nodes for explicit init."

        self.connections = {} if connections is None else connections
        self.nodes = {} if nodes is None else nodes
        self.population = population
        self.fitness = 0.0 # Fitness is evaluated externally
        self.config = config
        # Use config shape if available, otherwise use provided shape
        self.shape = config.genome_shape if config is not None else shape

        # If default initialization is requested, create the minimal structure
        if default_init:
            self.initialize(self.shape) # Initialize with the determined shape

    def copy(self):
        """
        Creates and returns a deep copy of this genome.

        The copy includes new Node and Connection objects, preserving the structure,
        weights, and enabled statuses. Node references within connections are updated
        to point to the copied nodes.

        Returns
        -------
        Genome
            A new Genome object that is a deep copy of this one.
        """
        # Copy all nodes first
        new_nodes = {node_id: node.copy() for node_id, node in self.nodes.items()}
        new_conns = {}
        # Copy all connections, ensuring they reference the *new* copied nodes
        for conn_id, conn in self.connections.items():
            # Check if both endpoints of the connection exist in the copied nodes
            if conn.in_node.id in new_nodes and conn.out_node.id in new_nodes:
                new_conn = conn.copy()
                # Update node references to point to the copied nodes
                new_conn.in_node = new_nodes.get(conn.in_node.id)
                new_conn.out_node = new_nodes.get(conn.out_node.id)
                new_conns[conn_id] = new_conn
            # else: connection involves a node not present in self.nodes, skip copying it
            # This case should ideally not happen if genome structure is consistent.

        # Create the new Genome instance with copied components
        copied_genome = Genome(self.population, shape=(1, 1), connections=new_conns, nodes=new_nodes, config=self.config)
        copied_genome.fitness = self.fitness # Copy fitness as well
        copied_genome.shape = self.shape # Copy shape
        return copied_genome

    @property
    def sorted_conns(self):
        """
        Returns a list of the genome's connections, sorted by innovation ID.

        Returns
        -------
        list[Connection]
            Sorted list of Connection objects.
        """
        return sorted(self.connections.values(), key=lambda c: c.id)

    @property
    def sorted_nodes(self):
        """
        Returns a list of the genome's nodes, sorted by node ID.

        Returns
        -------
        list[Node]
            Sorted list of Node objects.
        """
        return sorted(self.nodes.values(), key=lambda n: n.id)

    def add_node(self, connection_id):
        """
        Performs the "add node" mutation.

        Splits an existing connection by disabling it and inserting a new node
        in its place. Two new connections are created: one from the original
        input node to the new node, and one from the new node to the original
        output node.

        Parameters
        ----------
        connection_id : int
            The innovation ID of the connection to split.
        """
        old_conn = self.connections.get(connection_id)
        if old_conn is None:
            # Connection ID might not exist if it was removed or never added
            # This could happen e.g. if the chosen connection comes from a parent
            # but wasn't inherited during crossover. Handle gracefully.
            print(f"Warning: Attempted to add node on non-existent connection ID {connection_id}. Mutation skipped.")
            return

        # Disable the old connection
        old_conn.enabled = False

        # Get the original input and output nodes
        in_node = old_conn.in_node
        out_node = old_conn.out_node

        # Get a new unique node ID from the population tracker
        new_node_id = self.population.set_node_id()

        # Determine activation function based on config or default
        hid_activation = self.config.hid_node_activation if self.config is not None else 'relu'
        # Create the new hidden node
        node = Node('hidden', new_node_id, activation=hid_activation)
        self.nodes[node.id] = node # Add the new node to the genome

        # Create the connection from the original input node to the new node
        # Get a potentially new innovation ID for this connection topology
        conn1_id = self.population.set_conn_id(in_node.id, node.id)
        conn1 = Connection(in_node, node, conn1_id)
        conn1.weight = 1.0 # Standard practice: weight of input->new_node is 1.0
        self.connections[conn1.id] = conn1 # Add the first new connection

        # Create the connection from the new node to the original output node
        # Get a potentially new innovation ID for this connection topology
        conn2_id = self.population.set_conn_id(node.id, out_node.id)
        conn2 = Connection(node, out_node, conn2_id)
        conn2.weight = old_conn.weight # Standard practice: weight of new_node->output inherits old weight
        self.connections[conn2.id] = conn2 # Add the second new connection

    def check_connection(self, in_node_id, out_node_id):
        """
        Checks if a direct connection already exists between two specified nodes.

        Parameters
        ----------
        in_node_id : int
            The ID of the potential input node.
        out_node_id : int
            The ID of the potential output node.

        Returns
        -------
        bool
            True if a connection from `in_node_id` to `out_node_id` exists
            in `self.connections`, False otherwise.
        """
        for conn in self.connections.values():
            if conn.in_node.id == in_node_id and conn.out_node.id == out_node_id:
                return True
        return False

    def check_node(self, node_id):
        """
        Checks if a node with the given ID exists in the genome's node set.

        Parameters
        ----------
        node_id : int
            The ID of the node to check for.

        Returns
        -------
        bool
            True if a node with the specified ID exists in `self.nodes`, False otherwise.
        """
        return node_id in self.nodes

    def check_cycle(self, in_node_id, out_node_id):
        """
        Checks if adding a directed connection from `in_node_id` to `out_node_id`
        would create a cycle in the network graph.

        Uses Depth First Search (DFS) on the graph represented by the *enabled* connections,
        temporarily including the potential new connection.

        Parameters
        ----------
        in_node_id : int
            The ID of the starting node for the potential new connection.
        out_node_id : int
            The ID of the ending node for the potential new connection.

        Returns
        -------
        bool
            True if adding the connection would create a cycle, False otherwise.
        """
        # Create temporary nodes if they don't exist (relevant if nodes were added but connection not yet)
        # Note: This check assumes the nodes *will* exist if the connection is added.
        # A more robust check might involve only existing nodes or handling node existence differently.
        in_node = self.nodes.get(in_node_id) or Node('hidden', in_node_id) # Temporary representation if needed
        out_node = self.nodes.get(out_node_id) or Node('hidden', out_node_id) # Temporary representation if needed

        # Create a temporary connection representation
        temp_conn = Connection(in_node, out_node, id=float('inf')) # Use infinite ID to avoid collision

        # Build a temporary graph structure including the potential connection
        temp_connections = dict(self.connections)
        temp_connections[temp_conn.id] = temp_conn

        temp_nodes = dict(self.nodes)
        if not self.check_node(in_node.id): temp_nodes[in_node.id] = in_node # Add temp node if not present
        if not self.check_node(out_node.id): temp_nodes[out_node.id] = out_node # Add temp node if not present

        # Initialize DFS structures
        num_nodes = len(temp_nodes)
        visited = {node_id: False for node_id in temp_nodes} # Track visited nodes in current DFS path
        rec_stack = {node_id: False for node_id in temp_nodes} # Track nodes currently in recursion stack

        # Build adjacency list (graph representation: node -> list of nodes pointing *to* it)
        # We traverse backwards from a node to detect cycles efficiently with DFS.
        graph = {node_id: [] for node_id in temp_nodes}
        for conn in temp_connections.values():
            # Only consider enabled connections for cycle detection in the active network
             if conn.enabled:
                # Ensure both nodes exist in our temporary node set before adding edge
                if conn.in_node.id in graph and conn.out_node.id in graph:
                    # Add edge: in_node points to out_node -> store in_node as predecessor of out_node
                    graph[conn.out_node.id].append(conn.in_node.id)

        # DFS function to detect cycles
        def dfs(node_id):
            visited[node_id] = True
            rec_stack[node_id] = True

            # Recur for all predecessors (neighbors in the reversed graph)
            for neighbor in graph.get(node_id, []): # Use .get for safety if node somehow has no entry
                if neighbor in visited: # Check if neighbor exists in visited dict
                    if not visited[neighbor]:
                        if dfs(neighbor): # If recursive call finds a cycle
                            return True
                    elif rec_stack[neighbor]: # If neighbor is already in recursion stack -> cycle detected
                        return True
                # else: neighbor node ID not in temp_nodes, shouldn't happen with current setup

            # Remove node from recursion stack as we backtrack
            rec_stack[node_id] = False
            return False # No cycle found starting from this node in this path

        # Run DFS from all nodes to handle disconnected graphs
        for node_id in temp_nodes:
            if node_id in visited and not visited[node_id]: # Check if node_id exists in visited before accessing
                if dfs(node_id):
                    return True # Cycle detected

        return False # No cycles found

    def add_connection(self, in_node_id, out_node_id):
        """
        Adds a new connection gene between two existing nodes, if valid.

        Checks for several conditions before adding:
        1. The nodes must exist.
        2. The output node cannot be an input node.
        3. The input node cannot be an output node.
        4. The connection doesn't already exist.
        5. A connection in the opposite direction doesn't exist.
        6. Adding the connection doesn't create a cycle.

        Parameters
        ----------
        in_node_id : int
            The ID of the node where the connection starts.
        out_node_id : int
            The ID of the node where the connection ends.

        Returns
        -------
        bool
            True if the connection was successfully added, False otherwise (due to
            invalid conditions like cycle creation, existing connection or invalid node types).
        """
        # Get node objects (assume they exist if IDs are passed)
        in_node = self.nodes.get(in_node_id)
        out_node = self.nodes.get(out_node_id)

        # 1. If either node doesn't exist in the genome, cannot add connection
        if in_node is None or out_node is None:
            # print(f"Skipped adding connection {in_node_id}->{out_node_id}: Node(s) not found.")
            return False

        # 2. Output node cannot be an input node type
        if out_node.type == 'input':
            # print(f"Skipped adding connection {in_node_id}->{out_node_id}: Output node is input type.")
            return False
        # 3. Input node cannot be an output node type (for feedforward)
        if in_node.type == 'output':
            # print(f"Skipped adding connection {in_node_id}->{out_node_id}: Input node is output type.")
            return False

        # 4. Check if connection already exists
        if self.check_connection(in_node_id, out_node_id):
            # print(f"Skipped adding connection {in_node_id}->{out_node_id}: Already exists.")
            return False
        # 5. Check for connection in opposite direction (for feedforward networks)
        if self.check_connection(out_node_id, in_node_id):
            # print(f"Skipped adding connection {in_node_id}->{out_node_id}: Opposite exists.")
            return False

        # 6. Check for cycles
        if self.check_cycle(in_node_id, out_node_id):
            # print(f"Skipped adding connection {in_node_id}->{out_node_id}: Creates cycle.")
            return False

        # All checks passed, add the connection
        # Get the global innovation ID for this new connection topology
        conn_id = self.population.set_conn_id(in_node.id, out_node.id)
        conn = Connection(in_node, out_node, conn_id)
        # Weight is initialized randomly inside Connection.__init__
        self.connections[conn.id] = conn

        # print(f"Added connection {conn_id}: {in_node_id}->{out_node_id}")
        return True

    def add_node_mutation(self, prob=0.05):
        """
        Applies the "add node" mutation to the genome with a given probability.

        Randomly selects an enabled connection to split.

        Parameters
        ----------
        prob : float, optional
            The probability of attempting this mutation. Defaults to 0.05.
            If `self.config` is set, `self.config.add_node_mutation_prob` is used instead.
        """
        # Use probability from config if available
        prob = self.config.add_node_mutation_prob if self.config is not None else prob

        if random() < prob:
            # Find connections eligible for splitting (must be enabled)
            enabled_conns = [c for c in self.connections.values() if c.enabled]
            if not enabled_conns:
                # No enabled connections to split, mutation cannot occur
                return

            # Choose one enabled connection randomly
            conn_to_split = choice(enabled_conns)
            # Perform the add node operation
            self.add_node(conn_to_split.id)

    def add_connection_mutation(self, prob=0.08):
        """
        Applies the "add connection" mutation to the genome with a given probability.

        Attempts to add a new valid connection between two existing nodes that
        are not already directly connected. Retries a fixed number of times if
        the initially chosen pair is invalid.

        Parameters
        ----------
        prob : float, optional
            The probability of attempting this mutation. Defaults to 0.08.
            If `self.config` is set, `self.config.add_conn_mutation_prob` is used instead.
        """
        # Use probability from config if available
        prob = self.config.add_conn_mutation_prob if self.config is not None else prob

        if random() < prob:
            # Need at least two nodes to potentially add a connection
            if len(self.nodes) < 2:
                return

            # Allow multiple attempts to find a valid pair of nodes
            max_try = 35 # Limit attempts to avoid infinite loops in dense graphs
            for _ in range(max_try):
                # Ensure there are nodes to choose from
                if not self.nodes: return

                # Randomly select two distinct nodes
                node1 = choice(list(self.nodes.values()))
                node2 = choice(list(self.nodes.values()))

                # Ensure nodes are different
                if node1.id == node2.id:
                    continue

                # Try adding connection n1 -> n2
                if self.add_connection(node1.id, node2.id):
                    break # Success, exit loop
                # If n1 -> n2 failed (e.g., creates cycle, exists), try n2 -> n1
                elif self.add_connection(node2.id, node1.id):
                    break # Success, exit loop
                # If both directions failed, continue to next attempt

            # If loop finishes without adding, max_try was reached or no valid connections possible

    def weight_mutation(self, sigma=0.1, perturb_prob=0.8, reset_prob=0.1, min_weight=-1.0, max_weight=1.0):
        """
        Applies weight mutations to the connections in the genome.

        Each connection has a chance (`perturb_prob`) to be mutated. If selected,
        it has a further chance (`reset_prob`) to have its weight completely reset
        to a new random value. Otherwise, its weight is perturbed by adding
        Gaussian noise (mean 0, std dev `sigma`). Weights are clamped to [min_weight, max_weight].

        Parameters
        ----------
        sigma : float, optional
            Standard deviation for Gaussian noise perturbation. Defaults to 0.1.
            Overridden by `self.config.sigma` if available.
        perturb_prob : float, optional
            Probability of perturbing a connection's weight. Defaults to 0.8.
            Overridden by `self.config.perturb_prob` if available.
        reset_prob : float, optional
            Probability of resetting the weight (given perturbation occurs). Defaults to 0.1.
            Overridden by `self.config.reset_prob` if available.
        max_weight : float, optional
            Max weight reachable by the connection. Defaults to 1.0.
        min_weight : float, optional
            Min weight reachable by the connection. Defaults to -1.0.
        """
        # Use parameters from config if available
        sigma = self.config.sigma if self.config is not None else sigma
        perturb_prob = self.config.perturb_prob if self.config is not None else perturb_prob
        reset_prob = self.config.reset_prob if self.config is not None else reset_prob
        min_weight = self.config.min_weight if self.config is not None else min_weight
        max_weight = self.config.max_weight if self.config is not None else max_weight

        # Iterate through all connections in the genome
        for conn in self.connections.values():
            # Check if this connection should be perturbed
            if random() < perturb_prob:
                # Check if the weight should be reset or perturbed with noise
                if random() < reset_prob:
                    # Reset weight to a new random value in [-1, 1]
                    conn.weight = uniform(-1, 1)
                else:
                    # Perturb weight by adding Gaussian noise
                    conn.weight += gauss(0, sigma)
                # Clamp the weight to stay within the [-1, 1] range
                conn.weight = max(min_weight, min(max_weight, conn.weight))

    def mutate(self):
        """
        Applies all types of mutations (weight, add node, add connection)
        to the genome based on their respective probabilities defined in the config
        or default values.
        """
        # Order can matter slightly, e.g., mutating weights before structure changes
        # is common, but either order is acceptable.
        self.weight_mutation()
        self.add_node_mutation()
        self.add_connection_mutation()

    def topological_sort(self):
        """
        Performs a topological sort of the genome's nodes based on enabled connections.

        Uses Kahn's algorithm (based on in-degrees) to find a linear ordering
        of nodes such that for every directed edge from node u to node v,
        u comes before v in the ordering. Required for correct feed-forward activation.

        Returns
        -------
        list[int]
            A list of node IDs in topologically sorted order.

        Raises
        ------
        ValueError
            If the graph contains a cycle (among enabled connections), making
            topological sort impossible.
        """
        # Map from node ID to list of its *successors* (nodes it points to)
        adj_map = {node_id: [] for node_id in self.nodes}
        # Map from node ID to its *in-degree* (number of incoming enabled connections)
        in_degree = {node_id: 0 for node_id in self.nodes}

        # Populate adjacency map and in-degrees based on *enabled* connections
        for conn in self.connections.values():
             if conn.enabled:
                # Ensure both nodes are part of the graph (should always be true)
                if conn.in_node.id in in_degree and conn.out_node.id in in_degree:
                    # Add edge to adjacency map (if not already added for this pair)
                    # We need successors later for processing the queue.
                    if conn.out_node.id not in adj_map[conn.in_node.id]:
                         adj_map[conn.in_node.id].append(conn.out_node.id)
                    # Increment in-degree of the destination node
                    in_degree[conn.out_node.id] += 1

        # Initialize the queue with all nodes having an in-degree of 0
        # Sort initial queue by node ID for deterministic order
        queue = sorted([node_id for node_id, degree in in_degree.items() if degree == 0])
        sorted_nodes_ids = [] # This will store the topologically sorted order

        # Process nodes from the queue
        while queue:
            # Dequeue a node (take the first one)
            node_id = queue.pop(0)
            sorted_nodes_ids.append(node_id) # Add it to the sorted list

            # For each neighbor (successor) of the dequeued node
            # Sort neighbors by ID for deterministic processing
            neighbors = sorted(adj_map.get(node_id, []))
            for adj_node_id in neighbors:
                 if adj_node_id in in_degree: # Ensure neighbor is valid
                    # Decrease the in-degree of the neighbor
                    in_degree[adj_node_id] -= 1
                    # If in-degree becomes 0, enqueue the neighbor
                    if in_degree[adj_node_id] == 0:
                        queue.append(adj_node_id)
            # Keep the queue sorted by ID for consistent processing order
            queue.sort()

        # Check if a valid topological sort was found
        if len(sorted_nodes_ids) != len(self.nodes):
             # If the sorted list size doesn't match total nodes, there was a cycle
             raise ValueError("Graph contains a cycle, topological sort not possible.")

        return sorted_nodes_ids

    def forward(self, x):
        """
        Performs a feed-forward pass through the neural network defined by the genome.

        Activates nodes in topological order.

        Parameters
        ----------
        x : list[float] or list[int]
            A list of input values, corresponding to the input nodes in order.
            The length must match the number of input nodes in the genome.

        Returns
        -------
        list[float]
            A list of output values from the output nodes, in order of their IDs.
            Returns a list of zeros if a cycle is detected during topological sort.

        Raises
        ------
        ValueError
            If the input vector `x` size does not match the number of input nodes.
        """
        # Reset activation values for all nodes before the pass
        for node in self.nodes.values():
            node.value = 0.0

        # Get the topological order of nodes for activation
        try:
            sorted_node_ids = self.topological_sort()
        except ValueError as e:
            # Handle cycle detection: return default output
            print(f"Forward pass failed: {e}. Network likely has a cycle.")
            num_output_nodes = len([n for n in self.nodes.values() if n.type == 'output'])
            return [0.0] * num_output_nodes # Return default output

        # Assign input values to input nodes
        input_nodes = [node for node in self.nodes.values() if node.type == 'input']
        if len(x) != len(input_nodes):
            raise ValueError(f"Input vector size {len(x)} does not match number of input nodes {len(input_nodes)}")

        # Sort input nodes by ID to ensure consistent assignment
        input_nodes.sort(key=lambda n: n.id)
        for i, node in enumerate(input_nodes):
            # Check if the input node still exists in the genome (safety check)
            if node.id in self.nodes:
                 self.nodes[node.id].value = float(x[i]) # Assign input value

        # Precompute connection lookups for efficiency
        conn_map = {(conn.in_node.id, conn.out_node.id): conn
                    for conn in self.connections.values() if conn.enabled}
        # Precompute predecessors for each node (nodes that feed into it via enabled connections)
        pred_map = {node_id: [] for node_id in self.nodes}
        for conn in self.connections.values():
            if conn.enabled and conn.in_node.id in self.nodes and conn.out_node.id in self.nodes:
                 pred_map[conn.out_node.id].append(conn.in_node.id)

        # Activate nodes in topological order
        for node_id in sorted_node_ids:
            node = self.nodes.get(node_id)
            # Skip if node doesn't exist (shouldn't happen with check) or if it's an input node (already set)
            if node and node.type != 'input':
                # Calculate weighted sum from predecessors
                weighted_sum = 0.0
                predecessors = pred_map.get(node_id, [])
                for pred_node_id in predecessors:
                    conn = conn_map.get((pred_node_id, node_id)) # Look up connection efficiently
                    # If connection exists (should, if in pred_map) and predecessor exists...
                    if conn:
                        pred_node = self.nodes.get(pred_node_id)
                        if pred_node: # Safety check for predecessor node
                             weighted_sum += pred_node.value * conn.weight
                        # else: print warning about missing predecessor ? I will see in next reviews if it is important atm.

                # Apply activation function if defined
                if node.activation:
                     # Select the appropriate activation function
                     activation_function = select_activation(node.activation)
                     # Compute and store the node's output value
                     node.value = float(activation_function(weighted_sum))
                else:
                     # No activation function
                     # Currently, only hidden/output nodes are processed here, and they should have activations.
                     # If nodes without activation are possible, this handles them.
                     node.value = float(weighted_sum)

        # Collect output values from output nodes
        output_values = []
        output_nodes = [node for node in self.nodes.values() if node.type == 'output']
        # Sort output nodes by ID for consistent output order
        output_nodes.sort(key=lambda n: n.id)
        for node in output_nodes:
            output_values.append(node.value)

        return output_values

    def initialize(self, shape, activation='sigmoid'):
        """
        Initializes a minimal genome structure with input and output nodes
        and fully connected connections between them.

        Called by `__init__` when `connections` and `nodes` are not provided.

        Parameters
        ----------
        shape : tuple[int, int]
            The (number_of_inputs, number_of_outputs) for the network.
        activation : str, optional
            The name of the activation function for the output nodes. Defaults to 'sigmoid'.
            Overridden by `self.config.out_node_activation` if available.
        """
        # Use activation from config if available
        activation = self.config.out_node_activation if self.config is not None else activation

        # Reset nodes and connections dictionaries
        self.nodes = {}
        self.connections = {}
        in_nodes = []
        out_nodes = []

        # Create input nodes (IDs 0 to shape[0]-1)
        for i in range(shape[0]):
            # Input nodes typically have no activation function
            in_node = Node('input', i, activation=None)
            in_nodes.append(in_node)
            self.nodes[in_node.id] = in_node

        # Create output nodes (IDs shape[0] to shape[0]+shape[1]-1)
        for j in range(shape[1]):
            out_node_id = shape[0] + j
            out_node = Node('output', out_node_id, activation=activation)
            out_nodes.append(out_node)
            self.nodes[out_node_id] = out_node

        # Create initial connections: fully connect input layer to output layer
        for in_node in in_nodes:
            for out_node in out_nodes:
                # Get a unique innovation ID for this connection from the population tracker
                conn_id = self.population.set_conn_id(in_node.id, out_node.id)
                conn = Connection(in_node, out_node, conn_id)
                # Weight is initialized randomly in Connection.__init__
                self.connections[conn.id] = conn

        # Update the population's global node ID counter to avoid reusing IDs
        self.population.node_id = max(self.population.node_id, shape[0] + shape[1] - 1)

    def print_graph(self):
        """
        Prints a simple text representation of the genome's structure (nodes and connections)
        and its current fitness. Useful for debugging.
        """
        print("--- Genome Graph ---")
        print(f"Fitness: {self.fitness:.4f}") # Display fitness

        print("\nNodes:")
        if not self.nodes:
            print("  (No nodes defined)")
        else:
            # Print nodes sorted by ID for clarity
            for node in self.sorted_nodes:
                print(f"  {node}")

        print("\nConnections:")
        if not self.connections:
            print("  (No connections defined)")
        else:
            # Print connections sorted by innovation ID for clarity
            for conn in self.sorted_conns:
                print(f"  {conn}")

        print("--------------------\n")

class Species:
    """
    Represents a species in the population.

    A species groups genetically similar genomes together. It has a representative
    genome against which new genomes are compared for compatibility. Species help
    protect innovation by allowing genomes to compete primarily within their niche.

    Attributes
    ----------
    config : NEATConfig or None
        A reference to the main configuration object.
    members : list[Genome]
        A list of Genome objects belonging to this species.
    threshold : float
        The compatibility threshold used for this species. Typically inherited
        from the global config.
    representative : Genome or None
        A genome chosen (the first one added)to represent the species for
        compatibility checks.
    """
    def __init__(self, threshold = 3.0, config=None):
        """
        Initializes a new, empty species.

        Parameters
        ----------
        threshold : float, optional
            The compatibility threshold. Defaults to 3.0. Overridden by
            `config.species_threshold` if `config` is provided.
        config : NEATConfig, optional
            The configuration object. Defaults to None.
        """
        self.config = config
        self.members = [] # Start with no members
        # Use threshold from config if available, otherwise use provided default
        self.threshold = config.species_threshold if config is not None else threshold
        self.representative = None # Representative is set when the first member is added

    def adjust_fitness(self):
        """
        Adjusts the fitness of each member genome using explicit fitness sharing.

        The raw fitness of each genome is divided by the number of members in
        the species. This prevents single large species from dominating the
        selection process.

        Returns
        -------
        list[Genome]
            The list of members with adjusted fitness values. Returns an empty
            list if the species has no members.
        """
        num_members = len(self.members)
        if num_members == 0:
            return [] # No members, nothing to adjust

        # Divide each member's fitness by the species size
        for genome in self.members:
             genome.fitness /= float(num_members)

        return self.members

    def linear_scale_fitness(self, c=1.5):
        """
        Applies linear scaling to the fitness scores within the species.

        Scales fitness values to potentially amplify differences between individuals,
        aiming to increase selection pressure towards higher-performing members.
        Formula: f' = a*f + b, where a and b are chosen such that f'_avg = f_avg
        and f'_max = c * f_avg.

        Parameters
        ----------
        c : float, optional
            Scaling factor determining how much the maximum fitness should exceed
            the average fitness after scaling. Defaults to 1.5.

        Returns
        -------
        list[Genome]
            The list of members with linearly scaled fitness values.
        """
        # TODO: Consider adding the scaling factor 'c' to NEATConfig
        if not self.members:
            return []

        fitnesses = [g.fitness for g in self.members]
        f_avg = sum(fitnesses) / len(fitnesses)
        f_max = max(fitnesses)

        # Handle edge case where all fitnesses are equal
        if f_max == f_avg:
            # Avoid division by zero. Assign a default scaled fitness (e.g., 1.0)
            # or simply leave fitness as is if scaling isn't meaningful here.
            for g in self.members:
                g.fitness = 1.0 # Assign uniform scaled fitness
            return self.members

        # Calculate scaling parameters a and b
        a = (c - 1.0) * f_avg / (f_max - f_avg)
        b = f_avg * (1.0 - a)

        # Apply the linear scaling transformation
        for g in self.members:
            g.fitness = a * g.fitness + b

        return self.members

    def offset_fitness(self):
        """
        Offsets fitness scores within the species so that the minimum fitness is slightly above zero.

        This is often done before fitness proportionate selection (like roulette wheel
        which is not implemented yet) to ensure all individuals have a chance of being
        selected, even those with low or negative original fitness scores (if scaling
        resulted in negatives).

        Returns
        -------
        list[Genome]
            The list of members with offset fitness values.
        """
        if not self.members:
            return []

        f_min = min(g.fitness for g in self.members)
        # Add a small epsilon to ensure minimum is strictly positive
        epsilon = 1e-7
        offset = -f_min + epsilon

        # Apply the offset to all members
        for genome in self.members:
            genome.fitness += offset

        return self.members

    def rank(self):
        """
        Sorts the members of the species by fitness in descending order.

        Returns
        -------
        list[Genome]
            A new list containing the species members sorted from highest fitness
            to lowest fitness.
        """
        # Sort members based on their fitness attribute, highest first
        return sorted(self.members, key=lambda g: g.fitness, reverse=True)

class Population:
    """
    Manages the entire population of genomes, including speciation, reproduction
    and tracking of innovation numbers.

    Attributes
    ----------
    config : NEATConfig or None
        The configuration object guiding the population dynamics.
    genome_shape : tuple[int, int]
        The (input_size, output_size) used for initializing new genomes.
    size : int
        The target size of the population.
    species : list[Species]
        A list containing all the Species objects in the current generation.
    conn_id : int
        The highest innovation number assigned to a connection gene so far.
        Used to assign unique IDs to new connections. Initialized considering initial genome.
    node_id : int
        The highest node ID assigned so far. Used to assign unique IDs to new nodes.
        Initialized considering initial genome.
    conn_genes : dict[tuple[int, int], int]
        A dictionary mapping (in_node_id, out_node_id) tuples to the innovation ID
        assigned to that specific connection topology. Ensures the same structural
        innovation gets the same ID throughout evolution.
    members : list[Genome]
        The list of all Genome objects in the current population. After speciation,
        these genomes are also references within the `species` list members.
    """
    def __init__(self, config=None, genome_shape=(1,1), size=100):
        """
        Initializes the population.

        Parameters
        ----------
        genome_shape : tuple[int, int], optional
            The (input, output) shape for initial genomes. Defaults to (1, 1).
            Overridden by `config.genome_shape` if `config` is provided.
        size : int, optional
            The target population size. Defaults to 100.
            Overridden by `config.population_size` if `config` is provided.
        config : NEATConfig, optional
            The configuration object. If provided, its settings are used.
            Defaults to None.
        """
        self.config = config
        # Determine shape and size from config or parameters
        self.genome_shape = config.genome_shape if config is not None else genome_shape
        self.size = config.population_size if config is not None else size
        self.species = [] # List of species in the population
        # Initialize innovation counters. Start conn_id at -1 so first ID is 0.
        self.conn_id = -1
        # node_id starts after the highest ID used in the initial minimal genomes
        self.node_id = sum(self.genome_shape) - 1
        self.conn_genes = {} # Tracks innovation IDs for connections: {(in_id, out_id): innovation_id}
        self.members = [] # List to hold all genomes in the population

        # Create the initial population
        self.initialize(self.genome_shape)

    def set_conn_id(self, in_node_id, out_node_id):
        """
        Gets or assigns a unique innovation ID for a connection topology.

        Checks if a connection between the given input and output node IDs has
        been seen before. If yes, returns the existing innovation ID. If no,
        assigns a new innovation ID, stores it, and returns it.

        Parameters
        ----------
        in_node_id : int
            The ID of the input node for the connection.
        out_node_id : int
            The ID of the output node for the connection.

        Returns
        -------
        int
            The unique innovation ID for this connection topology.
        """
        key = (in_node_id, out_node_id)
        existing_id = self.conn_genes.get(key)

        if existing_id is not None:
            # This connection topology has been seen before, return its ID
            return existing_id
        else:
            # New connection topology, assign a new ID
            self.conn_id += 1 # Increment the global connection counter
            self.conn_genes[key] = self.conn_id # Store the mapping
            return self.conn_id

    def set_node_id(self):
        """
        Assigns and returns a new unique node ID.

        Increments the global node ID counter.

        Returns
        -------
        int
            A new, unique node ID.
        """
        self.node_id += 1
        return self.node_id

    def initialize(self, shape):
        """
        Creates the initial population of genomes and performs initial speciation.

        Generates `self.size` minimal genomes based on the provided `shape`,
        then assigns each genome to a species.

        Parameters
        ----------
        shape : tuple[int, int]
            The (input_size, output_size) for the initial genomes.
        """
        self.members = [] # Clear any existing members
        # Create the initial set of genomes
        for _ in range(self.size):
            # Each genome is created with the minimal structure
            genome = Genome(self, shape=shape, config=self.config)
            self.members.append(genome)

        # Reset species list and perform initial speciation
        self.species = []
        for genome in self.members:
            self.speciate(genome) # Assign each new genome to a species

    def categorize_genes(self, genome1, genome2):
        """
        Compares the connection genes of two genomes and categorizes them.

        Identifies matching genes (same innovation ID), disjoint genes (present in one
        genome within the ID range of the other but not matching), and excess genes
        (present in one genome beyond the ID range of the other). Used for calculating
        compatibility distance and for crossover.

        Parameters
        ----------
        genome1 : Genome
            The first genome to compare.
        genome2 : Genome
            The second genome to compare.

        Returns
        -------
        dict
            A dictionary containing categorized genes for both genomes:
            {
                'genome1': (matching, disjoint, excess),
                'genome2': (matching, disjoint, excess)
            }
            where each element in the tuples is a list of Connection objects.
            The 'matching' lists correspond to each other pair-wise.
        """
        # Get connections sorted by innovation ID for efficient comparison
        genes1 = genome1.sorted_conns
        genes2 = genome2.sorted_conns

        # Pointers for iterating through sorted gene lists
        idx1, idx2 = 0, 0
        # Lists to store categorized genes
        matching1, matching2 = [], [] # Matching genes from genome1 and genome2
        disjoint1, disjoint2 = [], [] # Disjoint genes from genome1 and genome2
        excess1, excess2 = [], []     # Excess genes from genome1 and genome2

        # Find the maximum innovation ID in each genome
        max_id1 = genes1[-1].id if genes1 else -1
        max_id2 = genes2[-1].id if genes2 else -1

        # Iterate through both gene lists simultaneously
        while idx1 < len(genes1) or idx2 < len(genes2):
            conn1 = genes1[idx1] if idx1 < len(genes1) else None # Current gene from genome1
            conn2 = genes2[idx2] if idx2 < len(genes2) else None # Current gene from genome2

            # Get innovation IDs, use infinity if one list is exhausted
            id1 = conn1.id if conn1 else float('inf')
            id2 = conn2.id if conn2 else float('inf')

            if id1 == id2: # Matching genes
                matching1.append(conn1)
                matching2.append(conn2)
                idx1 += 1
                idx2 += 1
            elif id1 < id2: # Gene exists in genome1 but not genome2 at this ID range
                # Check if it's excess (beyond genome2's max ID) or disjoint
                if id1 > max_id2:
                     excess1.append(conn1)
                else:
                     disjoint1.append(conn1)
                idx1 += 1
            elif id2 < id1: # Gene exists in genome2 but not genome1 at this ID range
                # Check if it's excess (beyond genome1's max ID) or disjoint
                if id2 > max_id1:
                    excess2.append(conn2)
                else:
                    disjoint2.append(conn2)
                idx2 += 1

        # Return the categorized lists
        return {
            'genome1': (matching1, disjoint1, excess1),
            'genome2': (matching2, disjoint2, excess2)
        }

    def cross_over(self, genome1, genome2):
        """
        Performs crossover between two parent genomes to create an offspring.

        The offspring inherits all genes from the more fit parent (genome1, by convention
        after potential swap). For matching genes (same innovation ID), the gene is
        chosen randomly from either parent. Genes only present in the more fit parent
        (disjoint and excess) are always inherited. Nodes are inherited from the more fit parent.

        Parameters
        ----------
        genome1 : Genome
            The first parent genome. Assumed to be the more fit parent (or equal fitness).
        genome2 : Genome
            The second parent genome.

        Returns
        -------
        Genome
            The newly created offspring genome. If crossover results in an invalid
            structure (e.g., cycle detection fails on the resulting structure, though
            this check is primarily during mutation), it might return a copy of the
            fitter parent as a fallback.
        """
        # Ensure genome1 is the fitter parent (or equal)
        if genome2.fitness > genome1.fitness:
            genome1, genome2 = genome2, genome1 # Swap if genome2 is fitter

        # Categorize genes from both parents
        categorized = self.categorize_genes(genome1, genome2)
        matching1, disjoint1, excess1 = categorized['genome1'] # Genes from fitter parent
        matching2, _, _ = categorized['genome2'] # Matching genes from less fit parent

        # Initialize offspring components
        # Start with all nodes from the fitter parent (genome1)
        offspring_nodes = {node.id: node.copy() for node in genome1.nodes.values()}
        offspring_connections = {}

        # Process matching genes: choose randomly from either parent
        for conn1, conn2 in zip(matching1, matching2):
            # Randomly select the connection gene from parent1 or parent2
            chosen_conn_gene_original = choice((conn1, conn2))
            chosen_conn_gene = chosen_conn_gene_original.copy() # Copy the chosen gene

            # Ensure the nodes for this connection exist in the offspring's node set
            if chosen_conn_gene.in_node.id in offspring_nodes and \
               chosen_conn_gene.out_node.id in offspring_nodes:
                # Update node references to point to the copied nodes in the offspring
                chosen_conn_gene.in_node = offspring_nodes.get(chosen_conn_gene.in_node.id)
                chosen_conn_gene.out_node = offspring_nodes.get(chosen_conn_gene.out_node.id)
                offspring_connections[chosen_conn_gene.id] = chosen_conn_gene

                # Handle disabled genes: If a gene is disabled in *either* parent,
                # there's a chance (e.g., 75%) it remains disabled in the offspring.
                if not conn1.enabled or not conn2.enabled:
                    if random() < 0.75: # Probability to inherit disabled status
                        chosen_conn_gene.enabled = False
                    # else: it might become enabled (inherited from the choice, or remains enabled)
            # else: If nodes don't exist (e.g. complex scenario after node deletion), skip this conn

        # Inherit disjoint and excess genes directly from the fitter parent (genome1)
        for conn in disjoint1 + excess1:
            new_conn = conn.copy() # Copy the gene
            # Ensure nodes exist and update references
            if new_conn.in_node.id in offspring_nodes and \
               new_conn.out_node.id in offspring_nodes:
                new_conn.in_node = offspring_nodes.get(new_conn.in_node.id)
                new_conn.out_node = offspring_nodes.get(new_conn.out_node.id)
                offspring_connections[new_conn.id] = new_conn
            # else: Skip if nodes don't exist

        # Create the offspring genome instance
        offspring = Genome(self, connections=offspring_connections, nodes=offspring_nodes, config=self.config)

        # Sanity check: Verify the offspring structure (e.g., for cycles)
        # This might be redundant if mutations handle cycle checks, but can catch crossover issues.
        try:
            offspring.topological_sort() # Check if sortable (no cycles)
        except Exception as e:
            # Fallback mechanism: if crossover creates an invalid structure,
            # return a copy of the fitter parent instead. This is a simple recovery.
            print(f"Error during crossover validation: {e}. Offspring generation failed. Returning copy of fitter parent.")
            offspring = genome1.copy()

        return offspring

    def calculate_compatibility(self, genome1, genome2, c1=1.0, c2=1.0, c3=0.4):
        """
        Calculates the compatibility distance (delta) between two genomes.

        The distance is a weighted sum of the number of excess genes (E),
        disjoint genes (D), and the average weight difference of matching genes (W).
        Formula: delta = (c1 * E / N) + (c2 * D / N) + (c3 * W)
        where N is the number of genes in the larger genome (normalizing factor).

        Parameters
        ----------
        genome1 : Genome
            The first genome.
        genome2 : Genome
            The second genome.
        c1 : float, optional
            Coefficient for excess genes. Defaults to 1.0. Overridden by `self.config.c1`.
        c2 : float, optional
            Coefficient for disjoint genes. Defaults to 1.0. Overridden by `self.config.c2`.
        c3 : float, optional
            Coefficient for average weight difference. Defaults to 0.4. Overridden by `self.config.c3`.

        Returns
        -------
        float
            The compatibility distance delta between the two genomes.
        """
        # Use coefficients from config if available
        c1 = self.config.c1 if self.config is not None else c1
        c2 = self.config.c2 if self.config is not None else c2
        c3 = self.config.c3 if self.config is not None else c3

        # Categorize genes to find E, D, and matching pairs
        categorized = self.categorize_genes(genome1, genome2)
        matching1, disjoint1, excess1 = categorized['genome1']
        matching2, disjoint2, excess2 = categorized['genome2']

        # Get the total number of connection genes in each genome
        n1 = len(genome1.connections) # Use full dict length, not just sorted list
        n2 = len(genome2.connections)

        # N is the number of genes in the larger genome (or 1 if both are empty)
        # Avoid division by zero if genomes are small (e.g., N=1 if max(n1,n2) < 20)
        N = max(1.0, float(max(n1, n2))) # Normalization factor

        # Count total excess and disjoint genes
        E = float(len(excess1) + len(excess2))
        D = float(len(disjoint1) + len(disjoint2))

        # Calculate average weight difference (W) for matching genes
        num_matching = len(matching1)
        if num_matching > 0:
            # Sum absolute differences in weights for all matching pairs
            weight_diff_sum = sum(abs(conn1.weight - conn2.weight) for conn1, conn2 in zip(matching1, matching2))
            W = weight_diff_sum / float(num_matching) # Average difference
        else:
            # If no matching genes, weight difference contribution is 0
            W = 0.0

        # Calculate the compatibility distance using the formula
        delta = (c1 * E / N) + (c2 * D / N) + (c3 * W)
        # print(f"Delta calculation: E={E}, D={D}, W={W:.4f}, N={N}, c1={c1}, c2={c2}, c3={c3} -> delta={delta:.4f}")
        return delta

    def speciate(self, genome):
        """
        Assigns a given genome to a species.

        Compares the genome to the representative of each existing species. If the
        compatibility distance is below the species threshold, the genome is added
        to that species. If it's not compatible with any existing species, a new
        species is created with this genome as its representative.

        Parameters
        ----------
        genome : Genome
            The genome to be assigned to a species.
        """
        assigned = False # Flag to track if the genome was assigned

        # If no species exist yet, create the first one
        if not self.species:
            new_species = Species(config=self.config) # Create species
            new_species.members.append(genome)
            new_species.representative = genome # This genome becomes the representative
            self.species.append(new_species)
            assigned = True
        else:
            # Iterate through existing species
            for species_obj in self.species:
                # Ensure the species has a representative (should always have one if not empty)
                if species_obj.representative is None:
                     # Attempt to set a representative if missing (e.g., first member)
                     if species_obj.members:
                          species_obj.representative = species_obj.members[0]
                     else:
                          # Skip empty species that might occur transiently
                          continue # Should ideally be cleaned up later

                # Calculate compatibility distance between the genome and the species representative
                delta = self.calculate_compatibility(species_obj.representative, genome)

                # If distance is below threshold, add genome to this species
                if delta < species_obj.threshold:
                    species_obj.members.append(genome)
                    assigned = True
                    break # Genome assigned, no need to check other species

            # If the genome wasn't assigned to any existing species, create a new one
            if not assigned:
                new_species = Species(config=self.config)
                new_species.members.append(genome)
                new_species.representative = genome # Becomes representative of the new species
                self.species.append(new_species)

    def reproduce(self, num_elites=1, selection_share=0.2):
        """
        Creates the next generation of genomes through selection, crossover and mutation.

        Steps:
        1. Adjust fitness scores within each species (scaling, offsetting, sharing).
        2. Calculate the number of offspring each species should produce based on its average adjusted fitness.
        3. Select parents from the top fraction (`selection_share`) of each species.
        4. Preserve elites (`num_elites`) by copying them directly to the next generation.
        5. Generate the remaining offspring through crossover between selected parents, followed by mutation.
        6. Replace the current population (`self.members`) with the new generation.
        7. Speciate the new population.

        Parameters
        ----------
        num_elites : int, optional
            Number of top genomes per species to carry over unchanged. Defaults to 1.
            Overridden by `self.config.num_elites`.
        selection_share : float, optional
            Fraction of top genomes per species eligible for mating. Defaults to 0.2.
            Overridden by `self.config.selection_share`.
        """
        # Use parameters from config if available
        num_elites = self.config.num_elites if self.config is not None else num_elites
        selection_share = self.config.selection_share if self.config is not None else selection_share

        new_pop = [] # List to hold the next generation of genomes
        species_data = [] # Stores data about each species for reproduction calculation

        # --- Fitness Adjustment and Offspring Allocation ---
        total_average_fitness = 0 # Sum of average adjusted fitness across all species
        # Filter out species that might have become empty
        living_species = [s for s in self.species if s.members]

        # Handle case where no species survived (useful for future more complex interactions with species).
        if not living_species:
             print("Warning: No living species found for reproduction. Repopulating randomly.")
             # Simple recovery: create a new random population
             while len(new_pop) < self.size:
                  new_pop.append(Genome(self, self.genome_shape, config=self.config))
             self.members = new_pop[:self.size] # Assign new population
             # Need to speciate this new random population
             self.species = []
             for genome in self.members:
                 self.speciate(genome)
             return # Exit reproduction phase

        # Process each living species: adjust fitness, rank members, calculate average fitness
        for species in living_species:
            # Apply fitness adjustments (order can matter)
            species.linear_scale_fitness() # Optional: Scaling before offsetting/sharing
            species.offset_fitness() # Optional: Ensure positive fitness for proportional selection
            species.adjust_fitness() # Apply fitness sharing based on species size
            ranked_members = species.rank() # Get members sorted by adjusted fitness
            # Calculate total and average fitness *after* adjustment
            species_total_fitness = sum(g.fitness for g in species.members)
            # Avoid division by zero if somehow a species exists but has 0 members after filtering
            species_average_fitness = species_total_fitness / len(species.members) if species.members else 0.0
            total_average_fitness += species_average_fitness
            # Store data for offspring calculation
            species_data.append({
                'species': species,
                'avg_fitness': species_average_fitness,
                'ranked': ranked_members
            })

        # Allocate offspring slots to species based on relative average fitness
        if total_average_fitness > 0:
            total_allocated = 0 # Track allocated offspring count
            for data in species_data:
                 # Proportion of total average fitness
                 proportion = data['avg_fitness'] / total_average_fitness
                 # Ensure proportion is not negative
                 proportion = max(0.0, proportion)
                 # Calculate number of offspring for this species (rounded)
                 data['num_offspring'] = int(round(proportion * self.size))
                 total_allocated += data['num_offspring']

            # Adjust offspring allocation to exactly match population size due to rounding
            discrepancy = self.size - total_allocated
            if discrepancy != 0 and species_data: # If there's a difference and species exist
                 # Sort species by offspring count to prioritize adding/removing from smaller/larger allocations
                 species_data.sort(key=lambda x: x['num_offspring'], reverse=(discrepancy > 0))
                 # Distribute or remove the difference one by one
                 for i in range(abs(discrepancy)):
                     idx_to_adjust = i % len(species_data) # Cycle through species
                     species_data[idx_to_adjust]['num_offspring'] += 1 if discrepancy > 0 else -1
                     # Ensure offspring count doesn't go below zero
                     species_data[idx_to_adjust]['num_offspring'] = max(0, species_data[idx_to_adjust]['num_offspring'])
        else:
            # Handle case where total average fitness is zero or negative
            print(f"Warning: Total average adjusted fitness ({total_average_fitness:.2f}) is zero or negative. Allocating offspring equally among {len(species_data)} species.")
            num_species = len(species_data)
            if num_species > 0:
                offspring_per_species = self.size // num_species
                remainder = self.size % num_species
                # Assign base number + distribute remainder
                for i, data in enumerate(species_data):
                    data['num_offspring'] = offspring_per_species + (1 if i < remainder else 0)
            # else: If species_data is empty (already handled above), this block is skipped

        # --- Generate New Population ---
        for data in species_data:
            species = data['species']
            num_offspring = data.get('num_offspring', 0) # Get allocated number
            ranked_members = data['ranked'] # Get sorted members

            # Skip if species gets 0 offspring or has no members
            if num_offspring == 0 or not ranked_members:
                continue

            # 1. Elitism: Copy top genomes directly
            elite_count = 0
            # Copy up to num_elites, but not more than available members or allocated offspring
            for i in range(min(num_elites, len(ranked_members), num_offspring)):
                 elite_copy = ranked_members[i].copy() # Make a copy
                 new_pop.append(elite_copy)
                 elite_count += 1

            # 2. Crossover and Mutation: Generate remaining offspring
            num_remaining_offspring = num_offspring - elite_count
            if num_remaining_offspring <= 0:
                 continue # All offspring slots filled by elites

            # Determine the pool of parents for crossover (top `selection_share` fraction)
            selection_pool_size = max(1, int(len(ranked_members) * selection_share))
            selection_pool = ranked_members[:selection_pool_size]

            # Check if selection pool is valid
            if not selection_pool:
                 # Fallback: If pool is empty
                 # Use the best member as parent or create random genomes if needed.
                 print(f"Warning: Selection pool empty for species {species.representative.id if species.representative else 'N/A'} despite {num_offspring} offspring needed. Using fallback.")
                 # Simple fallback: use the best available member if possible
                 parent_fallback = ranked_members[0] if ranked_members else None
                 for _ in range(num_remaining_offspring):
                     if parent_fallback:
                         # Create offspring from the single best parent (effectively asexual reproduction + mutation)
                         offspring = parent_fallback.copy()
                         offspring.mutate()
                         new_pop.append(offspring)
                     else:
                         # Extreme fallback: create random genome if no members exist
                         new_pop.append(Genome(self, self.genome_shape, config=self.config))
                 continue # Skip normal crossover loop

            # Generate offspring via crossover
            for _ in range(num_remaining_offspring):
                # Select two parents randomly from the selection pool
                parent1 = choice(selection_pool)
                parent2 = choice(selection_pool)

                # Perform crossover
                # Wrap in try-except for robustness against potential crossover errors
                max_try_crossover = 5 # Try crossover a few times if it fails initially
                offspring = None
                for attempt in range(max_try_crossover):
                    # TODO Improve consistency
                    try:
                        offspring = self.cross_over(parent1, parent2)
                        # If successful, break the attempt loop
                        break
                    except Exception as e: # Catch generic exceptions during crossover/validation
                        print(f'Error during crossover (Attempt {attempt+1}/{max_try_crossover}): {e}. Parents: P1(Fit:{parent1.fitness:.2f}), P2(Fit:{parent2.fitness:.2f})')
                        # If max attempts reached, use a copy of the fitter parent + mutation as fallback
                        if attempt == max_try_crossover - 1:
                            print("Crossover failed after retries. Using mutated copy of fitter parent.")
                            fitter_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
                            offspring = fitter_parent.copy()

                # Mutate the newly created offspring
                if offspring: # Ensure offspring was successfully created/assigned
                    offspring.mutate()
                    new_pop.append(offspring)
                else:
                    # Should not happen with the fallback, but as a safeguard:
                    print("ERROR: Offspring generation failed completely. Adding random genome.")
                    new_pop.append(Genome(self, self.genome_shape, config=self.config))


        # --- Finalize Population ---
        # Ensure population size is maintained
        while len(new_pop) < self.size:
            print(f"Population shortfall detected. Current size: {len(new_pop)}, Target: {self.size}. Adding random genome.")
            # Add new random genomes to fill the gap
            new_pop.append(Genome(self, self.genome_shape, config=self.config))

        # Replace the old population members with the new generation
        self.members = new_pop[:self.size] # Ensure exactly self.size members

        # --- Speciate the New Population ---
        # Clear old species assignments (keep representatives temporarily)
        for s in self.species:
            s.members = [] # Clear members list for each species

        # Assign the new members to species (potentially creating new ones)
        for genome in self.members:
            self.speciate(genome) # Assign genome to a species (finds compatible or creates new)

        # Clean up species: remove empty ones, reset representatives if needed
        self.species = [s for s in self.species if s.members] # Keep only species with members
        # Optionally, re-select representatives for remaining species (e.g., randomly from members)
        for s in self.species:
            if not s.representative or s.representative not in s.members:
                # If representative is gone or invalid, choose a new one randomly
                if s.members:
                    s.representative = choice(s.members)
                else:
                    s.representative = None

    def gather_population(self):
        """
        Collects all member genomes from all species into a single list.

        Returns
        -------
        list[Genome]
            A list containing all genomes currently assigned to species.
        """
        all_members = []
        for species in self.species:
            all_members.extend(species.members)
        return all_members

    def get_top_genome(self):
        """
        Finds and returns the genome with the highest fitness in the entire population.

        Returns
        -------
        Genome
            The genome object with the highest fitness score.

        Raises
        ------
        IndexError
            If the population is empty.
        """
        # Use self.members as the source of all current genomes
        if not self.members:
             raise IndexError("Cannot get top genome from an empty population.")
        # Sort all members by fitness (descending) and return the first one
        top_genome = sorted(self.members, key=lambda g: g.fitness, reverse=True)[0]
        return top_genome

    def save_top_genome(self, filename):
        """
        Saves the genome with the highest fitness in the current population to a file
        using pickle serialization.

        The save path is determined by `self.config.save_path`.

        Parameters
        ----------
        filename : str
            The base name for the output file (e.g., "best_genome_gen_10").
            ".pkl" extension will be added automatically.
        """
        # Determine the save directory from config or use current directory
        save_path = self.config.save_path if self.config is not None else "./"

        try:
            # Get the best genome from the current population
            genome_to_save = self.get_top_genome()
            # Construct the full file path
            full_path = f'{save_path}{filename}.pkl'
            # Open the file in binary write mode and dump the genome object
            with open(full_path, 'wb') as f:
                pkl.dump(genome_to_save, f)
            print(f"Top genome saved successfully to {full_path}")
        except IndexError:
            print("Could not save top genome: Population is empty.")
        except Exception as e:
            print(f"Error saving top genome to {filename}.pkl: {e}")

