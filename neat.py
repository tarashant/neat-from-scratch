from random import uniform, randint, choice, random, gauss
import pickle as pkl
import math

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def select_activation(name):
    if name == 'relu':
        return relu
    elif name == 'sigmoid':
        return sigmoid
    else:
        raise ValueError('Unknown activation function.')

class NEATConfig:
    def __init__(
            self,
            population_size=100,
            genome_shape=(1,1),
            hid_node_activation='relu',
            out_node_activation='sigmoid',
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
        self.genome_shape = genome_shape
        self.add_node_mutation_prob = add_node_mutation_prob
        self.add_conn_mutation_prob = add_conn_mutation_prob
        self.sigma = sigma
        self.perturb_prob = perturb_prob
        self.reset_prob = reset_prob
        self.hid_node_activation = hid_node_activation
        self.out_node_activation = out_node_activation
        self.species_threshold = species_threshold
        self.population_size = population_size
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_elites = num_elites
        self.selection_share = selection_share
        self.save_path = save_path

    def __repr__(self):
        str = ""
        for key, value in self.__dict__.items():
            str += f"{key}: {value}\n"
        return str

class Connection:
    def __init__(self, in_node, out_node, id=0):
        self.in_node = in_node
        self.out_node = out_node
        self.id = id
        self.weight = uniform(-1,1)
        self.enabled = True

    def copy(self):
        conn = Connection(self.in_node, self.out_node, self.id)
        conn.enabled = self.enabled
        conn.weight = self.weight
        return conn

    def __repr__(self):
        return f"id: {self.id}: nodes: {self.in_node.id} -> {self.out_node.id} (W:{self.weight:.2f} E:{self.enabled})"

class Node:
    def __init__(self, type, id, activation=None):
        self.type = type
        self.value = 0.0
        self.activation = activation
        self.id = id

    def copy(self):
        node = Node(self.type, self.id, self.activation)
        return node

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __repr__(self):
        return f"Node(id={self.id}, type={self.type}, act={self.activation})"

class Genome:
    def __init__(self, population, shape=(1,1), connections=None, nodes=None, config=None):
        default_init = ((shape != (1, 1) or (config.genome_shape != (1,1) if config is not None else False)) and connections is None and nodes is None)
        conn_node_init = ((shape == (1, 1) or (config.genome_shape == (1,1) if config is not None else False)) and connections is not None and nodes is not None)
        assert default_init or conn_node_init, \
            "Provide shape or connections and nodes."

        self.connections = {} if connections is None else connections
        self.nodes = {} if nodes is None else nodes
        self.population = population
        self.fitness = 0.0
        self.shape = config.genome_shape if config is not None else shape
        self.config = config

        if default_init:
            self.initialize(shape)

    def copy(self):
        new_nodes = {node_id: node.copy() for node_id, node in self.nodes.items()}
        new_conns = {}
        for conn_id, conn in self.connections.items():
            if conn.in_node.id in new_nodes and conn.out_node.id in new_nodes:
                new_conn = conn.copy()
                new_conn.in_node = new_nodes.get(conn.in_node.id)
                new_conn.out_node = new_nodes.get(conn.out_node.id)
                new_conns[conn_id] = new_conn
        return Genome(self.population, shape=(1, 1), connections=new_conns, nodes=new_nodes, config=self.config)

    @property
    def sorted_conns(self):
        return sorted(self.connections.values(), key=lambda c: c.id)

    @property
    def sorted_nodes(self):
        return sorted(self.nodes.values(), key=lambda n: n.id)

    def add_node(self, connection_id):
        old_conn = self.connections.get(connection_id)

        old_conn.enabled = False

        in_node = old_conn.in_node
        out_node = old_conn.out_node

        new_node_id = self.population.set_node_id()

        node = Node('hidden', new_node_id, activation=self.config.hid_node_activation if self.config is not None else 'relu')
        self.nodes[node.id] = node

        conn1_id = self.population.set_conn_id(in_node.id, node.id)
        conn1 = Connection(in_node, node, conn1_id)
        conn1.weight = 1.0
        self.connections[conn1.id] = conn1

        conn2_id = self.population.set_conn_id(node.id, out_node.id)
        conn2 = Connection(node, out_node, conn2_id)
        conn2.weight = old_conn.weight
        self.connections[conn2.id] = conn2

    def check_connection(self, in_node_id, out_node_id):
        for conn in self.connections.values():
            if conn.in_node.id == in_node_id and conn.out_node.id == out_node_id:
                return True
        return False

    def check_node(self, node_id):
        if node_id in self.nodes:
                return True
        return False

    def check_cycle(self, in_node_id, out_node_id):
        in_node = Node('hidden', in_node_id)
        out_node = Node('hidden', out_node_id)

        conn = Connection(in_node, out_node, id=float('inf'))
        connections = dict(self.connections)
        connections[conn.id] = conn

        nodes = dict(self.nodes)
        if not self.check_node(in_node.id):
            nodes[in_node.id] = in_node
        if not self.check_node(out_node.id):
            nodes[out_node.id] = out_node

        num_nodes = len(nodes)
        visited = {node.id: False for node in nodes.values()}
        rec_stack = {node.id: False for node in nodes.values()}

        graph = {node.id: [] for node in nodes.values()}
        for conn in connections.values():
             if conn.enabled:
                graph[conn.out_node.id].append(conn.in_node.id)

        def dfs(node_id):
            visited[node_id] = True
            rec_stack[node_id] = True
            for neighbor in graph[node_id]:
                if not visited[neighbor]:
                    if dfs(neighbor):
                        return True
                elif rec_stack[neighbor]:
                    return True
            rec_stack[node_id] = False
            return False

        for node_id in nodes:
            if not visited[node_id]:
                if dfs(node_id):
                    return True
        return False

    def add_connection(self, in_node_id, out_node_id):
        if self.check_cycle(in_node_id, out_node_id):
            return False

        in_node = self.nodes[in_node_id]
        out_node = self.nodes[out_node_id]

        if self.check_connection(in_node_id, out_node_id):
            return False
        if self.check_connection(out_node_id, in_node_id):
             return False
        if out_node.type == 'input':
             return False
        if in_node.type == 'output':
             return False

        conn_id = self.population.set_conn_id(in_node.id, out_node.id)
        conn = Connection(in_node, out_node, conn_id)
        self.connections[conn.id] = conn

        #print(f"Added connection {conn_id}: {in_node_id}->{out_node_id}")
        return True

    def add_node_mutation(self, prob=0.05):
        prob = self.config.add_node_mutation_prob if self.config is not None else prob
        if random() < prob:
            enabled_conns = [c for c in self.connections.values() if c.enabled]
            if not enabled_conns:
                return
            conn = choice(enabled_conns)
            self.add_node(conn.id)

    def add_connection_mutation(self, prob=0.08):
        prob = self.config.add_conn_mutation_prob if self.config is not None else prob
        if random() < prob:
            if len(self.nodes.values()) < 2:
                return

            max_try = 35
            for _ in range(max_try):
                if not self.nodes: return

                in_node = choice(list(self.nodes.values()))
                out_node = choice(list(self.nodes.values()))

                if in_node.id == out_node.id:
                    continue
                if out_node.type == 'input':
                    continue
                if in_node.type == 'output':
                    continue

                if self.add_connection(in_node.id, out_node.id):
                        break

    def weight_mutation(self, sigma=0.1, perturb_prob=0.8, reset_prob=0.1):
        sigma = self.config.sigma if self.config is not None else sigma
        perturb_prob = self.config.perturb_prob if self.config is not None else perturb_prob
        reset_prob = self.config.reset_prob if self.config is not None else reset_prob
        for conn in self.connections.values():
            if random() < perturb_prob:
                if random() < reset_prob:
                    conn.weight = uniform(-1, 1)
                else:
                    conn.weight += gauss(0, sigma)
                conn.weight = max(-1.0, min(1.0, conn.weight))

    def mutate(self):
        self.weight_mutation()
        self.add_node_mutation()
        self.add_connection_mutation()

    def topological_sort(self):
        adj_map = {node.id: [] for node in self.nodes.values()}
        in_degree = {node.id: 0 for node in self.nodes.values()}

        for conn in self.connections.values():
             if conn.enabled:
                if conn.in_node.id in in_degree and conn.out_node.id in in_degree:
                    if conn.out_node.id not in adj_map[conn.in_node.id]:
                         adj_map[conn.in_node.id].append(conn.out_node.id)
                    in_degree[conn.out_node.id] += 1

        queue = sorted([node_id for node_id in in_degree if in_degree[node_id] == 0])
        sorted_nodes_ids = []

        while queue:
            node_id = queue.pop(0)
            sorted_nodes_ids.append(node_id)

            neighbors = sorted(adj_map.get(node_id, []))
            for adj_node_id in neighbors:
                 if adj_node_id in in_degree:
                    in_degree[adj_node_id] -= 1
                    if in_degree[adj_node_id] == 0:
                        queue.append(adj_node_id)
            queue.sort()

        if len(sorted_nodes_ids) != len(self.nodes):
             raise ValueError("Graph contains a cycle, topological sort not possible.")

        return sorted_nodes_ids

    def forward(self, x):
        for node in self.nodes.values():
            node.value = 0.0

        try:
            sorted_node_ids = self.topological_sort()
        except ValueError as e:
            print(f"Forward pass failed: {e}")
            num_output_nodes = len([n for n in self.nodes.values() if n.type == 'output'])
            return [0.0] * num_output_nodes

        input_nodes = [node for node in self.nodes.values() if node.type == 'input']
        if len(x) != len(input_nodes):
            raise ValueError(f"Input vector size {len(x)} != number of input nodes {len(input_nodes)}")

        input_nodes.sort(key=lambda n: n.id)
        for i, node in enumerate(input_nodes):
            if node.id in self.nodes:
                 self.nodes[node.id].value = float(x[i])

        conn_map = {(conn.in_node.id, conn.out_node.id): conn for conn in self.connections.values()}
        pred_map = {node_id: [] for node_id in self.nodes}
        for conn in self.connections.values():
            if conn.enabled and conn.in_node.id in self.nodes and conn.out_node.id in self.nodes:
                 conn_map[(conn.in_node.id, conn.out_node.id)] = conn
                 pred_map[conn.out_node.id].append(conn.in_node.id)

        for node_id in sorted_node_ids:
            node = self.nodes.get(node_id)
            if node and node.type != 'input':
                weighted_sum = 0.0
                predecessors = pred_map.get(node_id, [])
                for pred_node_id in predecessors:
                    conn = conn_map.get((pred_node_id, node_id))
                    pred_node = self.nodes.get(pred_node_id)
                    weighted_sum += pred_node.value * conn.weight

                if node.activation:
                     activation_function = select_activation(node.activation)
                     node.value = float(activation_function(weighted_sum))
                else:
                     node.value = float(weighted_sum)

        output_values = []
        output_nodes = [node for node in self.nodes.values() if node.type == 'output']

        output_nodes.sort(key=lambda n: n.id)
        for node in output_nodes:
            output_values.append(node.value)

        return output_values

    def initialize(self, shape, activation='sigmoid'):
        activation = self.config.out_node_activation if self.config is not None else activation
        self.nodes = {}
        self.connections = {}
        in_nodes = []
        out_nodes = []

        for i in range(shape[0]):
            in_node = Node('input', i)
            in_nodes.append(in_node)
            self.nodes[in_node.id] = in_node

        for j in range(shape[1]):
            out_node_id = shape[0] + j
            out_node = Node('output', out_node_id, activation=activation)
            out_nodes.append(out_node)
            self.nodes[out_node_id] = out_node

        for in_node in in_nodes:
            for out_node in out_nodes:
                conn_id = self.population.set_conn_id(in_node.id, out_node.id)
                conn = Connection(in_node, out_node, conn_id)
                self.connections[conn.id] = conn

        self.population.node_id = max(self.population.node_id, shape[0] + shape[1] - 1)

    def print_graph(self):
        print("--- Genome Graph ---")
        print(f"Fitness: {self.fitness:.4f}")

        print("\nNodes:")
        if not self.nodes:
            print("  (No nodes defined)")
        else:
            for node in self.sorted_nodes:
                print(f"  {node}")

        print("\nConnections:")
        if not self.connections.values():
            print("  (No connections defined)")
        else:
            for conn in self.sorted_conns:
                print(f"  {conn}")

        print("--------------------\n")

class Species:
    def __init__(self, threshold = 3.0, config=None):
        self.config = config
        self.members = []
        self.threshold = config.species_threshold if config is not None else threshold
        self.representative = None

    def adjust_fitness(self):
        num_members = len(self.members)
        if num_members == 0:
            return []
        for genome in self.members:
            genome.fitness /= num_members
        return self.members

    def linear_scale_fitness(self, c=1.5):
        #TODO Add to config
        fitnesses = [g.fitness for g in self.members]
        f_avg = sum(fitnesses) / len(fitnesses)
        f_max = max(fitnesses)
        if f_max == f_avg:
            for g in self.members:
                g.fitness = 1.0
            return self.members
        a = (c - 1) * f_avg / (f_max - f_avg)
        b = f_avg * (1 - a)
        for g in self.members:
            g.fitness = a * g.fitness + b
        return self.members

    def offset_fitness(self):
        f_min = min(g.fitness for g in self.members)
        for genome in self.members:
            genome.fitness -= f_min + 1e-7
        return self.members

    def rank(self):
        return sorted(self.members, key=lambda g: g.fitness, reverse=True)

class Population:
    def __init__(self, genome_shape=(1,1), size=100, config=None):
        self.config = config
        self.genome_shape = config.genome_shape if config is not None else genome_shape
        self.size = config.population_size if config is not None else size
        self.species = []
        self.conn_id = -1
        self.node_id = sum(self.genome_shape) - 1
        self.conn_genes = {}
        self.members = []

        self.initialize(self.genome_shape)

    def set_conn_id(self, in_node_id, out_node_id):
        key = (in_node_id, out_node_id)
        existing_id = self.conn_genes.get(key)
        if existing_id is not None:
            return existing_id
        else:
            self.conn_id += 1
            self.conn_genes[key] = self.conn_id
            return self.conn_id

    def set_node_id(self):
        self.node_id += 1
        return self.node_id

    def initialize(self, shape):
        self.members = []
        for _ in range(self.size):
            genome = Genome(self, shape=shape, config=self.config)
            self.members.append(genome)

        self.species = []
        for genome in self.members:
            self.speciate(genome)

    def categorize_genes(self, genome1, genome2):
        genes1 = genome1.sorted_conns
        genes2 = genome2.sorted_conns

        idx1, idx2 = 0, 0
        matching1, matching2 = [], []
        disjoint1, disjoint2 = [], []
        excess1, excess2 = [], []

        max_id1 = genes1[-1].id if genes1 else -1
        max_id2 = genes2[-1].id if genes2 else -1

        while idx1 < len(genes1) or idx2 < len(genes2):
            conn1 = genes1[idx1] if idx1 < len(genes1) else None
            conn2 = genes2[idx2] if idx2 < len(genes2) else None

            id1 = conn1.id if conn1 else float('inf')
            id2 = conn2.id if conn2 else float('inf')

            if id1 == id2:
                matching1.append(conn1)
                matching2.append(conn2)
                idx1 += 1
                idx2 += 1
            elif id1 < id2:
                if id1 > max_id2:
                     excess1.append(conn1)
                else:
                     disjoint1.append(conn1)
                idx1 += 1
            elif id2 < id1:
                if id2 > max_id1:
                    excess2.append(conn2)
                else:
                    disjoint2.append(conn2)
                idx2 += 1

        return {
            'genome1': (matching1, disjoint1, excess1),
            'genome2': (matching2, disjoint2, excess2)
        }

    def cross_over(self, genome1, genome2):
        if genome2.fitness > genome1.fitness:
            genome1, genome2 = genome2, genome1  # Swap

        categorized = self.categorize_genes(genome1, genome2)
        matching1, disjoint1, excess1 = categorized['genome1']
        matching2, _, _ = categorized['genome2']

        offspring_connections = {}
        offspring_nodes = {node.id: node.copy() for node in genome1.nodes.values()}

        for conn1, conn2 in zip(matching1, matching2):
            chosen_conn_gene = choice((conn1, conn2)).copy()
            if chosen_conn_gene.in_node.id in offspring_nodes and \
                    chosen_conn_gene.out_node.id in offspring_nodes:
                chosen_conn_gene.in_node = offspring_nodes.get(chosen_conn_gene.in_node.id)
                chosen_conn_gene.out_node = offspring_nodes.get(chosen_conn_gene.out_node.id)
                offspring_connections[chosen_conn_gene.id] = chosen_conn_gene
                if not conn1.enabled or not conn2.enabled:
                    if random() < 0.75:
                        chosen_conn_gene.enabled = False

        for conn in disjoint1 + excess1:
            new_conn = conn.copy()
            if new_conn.in_node.id in offspring_nodes and \
                    new_conn.out_node.id in offspring_nodes:
                new_conn.in_node = offspring_nodes.get(new_conn.in_node.id)
                new_conn.out_node = offspring_nodes.get(new_conn.out_node.id)
                offspring_connections[new_conn.id] = new_conn

        offspring = Genome(self, connections=offspring_connections, nodes=offspring_nodes, config=self.config)

        try:
            offspring.topological_sort()
        except Exception as e:
            offspring = genome1.copy()
            print(f"Error in crossover: {e}")

        return offspring

    def calculate_compatibility(self, genome1, genome2, c1=1.0, c2=1.0, c3=0.4):
        c1 = self.config.c1 if self.config is not None else c1
        c2 = self.config.c2 if self.config is not None else c2
        c3 = self.config.c3 if self.config is not None else c3

        categorized = self.categorize_genes(genome1, genome2)
        matching1, disjoint1, excess1 = categorized['genome1']
        matching2, disjoint2, excess2 = categorized['genome2']

        n1 = len(genome1.connections.values())
        n2 = len(genome2.connections.values())

        N = float(max(n1, n2))

        E = float(len(excess1) + len(excess2))
        D = float(len(disjoint1) + len(disjoint2))

        num_matching = len(matching1)
        if num_matching > 0:
            weight_diff_sum = sum(abs(conn1.weight - conn2.weight) for conn1, conn2 in zip(matching1, matching2))
            W = weight_diff_sum / float(num_matching)
        else:
            W = 0.0

        delta = (c1 * E / N) + (c2 * D / N) + (c3 * W)
        return delta

    def speciate(self, genome):
        assigned = False
        if not self.species:
            new_species = Species(config=self.config)
            new_species.members.append(genome)
            new_species.representative = genome
            self.species.append(new_species)
            assigned = True
        else:
            for species_obj in self.species:
                if species_obj.representative is None:
                     if species_obj.members:
                          species_obj.representative = species_obj.members[0]
                     else:
                          continue

                delta = self.calculate_compatibility(species_obj.representative, genome)
                if delta < species_obj.threshold:
                    species_obj.members.append(genome)
                    assigned = True
                    break

            if not assigned:
                new_species = Species(config=self.config)
                new_species.members.append(genome)
                new_species.representative = genome
                self.species.append(new_species)

    def reproduce(self, num_elites=1, selection_share=0.2):
        # TODO Base the reproduction upon standardized fitnesses
        num_elites = self.config.num_elites if self.config is not None else num_elites
        selection_share = self.config.selection_share if self.config is not None else selection_share

        new_pop = []
        species_data = []

        total_average_fitness = 0
        living_species = [s for s in self.species if s.members]

        if not living_species:
             print("Warning: No living species to reproduce from. Repopulating randomly.")
             while len(new_pop) < self.size:
                  new_pop.append(Genome(self, self.genome_shape, config=self.config))
             self.members = new_pop[:self.size]
             return

        for species in living_species:
            species.linear_scale_fitness()
            species.offset_fitness()
            species.adjust_fitness()
            ranked_members = species.rank()
            species_total_fitness = sum(g.fitness for g in species.members)
            species_average_fitness = species_total_fitness / len(species.members)
            total_average_fitness += species_average_fitness
            species_data.append({'species': species, 'avg_fitness': species_average_fitness, 'ranked': ranked_members})

        if total_average_fitness > 0:
            total_allocated = 0
            for data in species_data:
                 proportion = data['avg_fitness'] / total_average_fitness
                 proportion = max(0, proportion)
                 data['num_offspring'] = int(round(proportion * self.size))
                 total_allocated += data['num_offspring']

            discrepancy = self.size - total_allocated
            if discrepancy != 0 and species_data:
                 species_data.sort(key=lambda x: x['num_offspring'], reverse=(discrepancy > 0))
                 for i in range(abs(discrepancy)):
                     idx_to_adjust = i % len(species_data)
                     species_data[idx_to_adjust]['num_offspring'] += 1 if discrepancy > 0 else -1
                     species_data[idx_to_adjust]['num_offspring'] = max(0, species_data[idx_to_adjust]['num_offspring'])
        else:
            print(f"Warning: Total average adjusted fitness ({total_average_fitness:.2f}) is zero or negative. Using equal allocation.")
            num_species = len(species_data)
            offspring_per_species = self.size // num_species
            remainder = self.size % num_species
            for i, data in enumerate(species_data):
                data['num_offspring'] = offspring_per_species + (1 if i < remainder else 0)

        for data in species_data:
            species = data['species']
            num_offspring = data.get('num_offspring', 0)
            ranked_members = data['ranked']

            if num_offspring == 0 or not ranked_members:
                continue

            elite_count = 0
            for i in range(min(num_elites, len(ranked_members), num_offspring)):
                 new_pop.append(ranked_members[i].copy())
                 elite_count += 1

            num_remaining_offspring = num_offspring - elite_count
            if num_remaining_offspring <= 0:
                 continue

            selection_pool_size = max(1, int(len(ranked_members) * selection_share))
            selection_pool = ranked_members[:selection_pool_size]

            if not selection_pool:
                 print(f"Warning: Selection pool empty for species despite {num_offspring} offspring needed.")
                 for _ in range(num_remaining_offspring):
                     new_pop.append(Genome(self, self.genome_shape, config=self.config))
                 continue

            for _ in range(num_remaining_offspring):
                parent1 = choice(selection_pool)
                parent2 = choice(selection_pool)
                max_try = 20
                for _ in range(max_try):
                    try:
                        offspring = self.cross_over(parent1, parent2)
                        break
                    except ValueError as e:
                        print(f'Error while crossing: {e}')
                offspring.mutate()
                new_pop.append(offspring)

        while len(new_pop) < self.size:
            print(f"Population shortfall. Current: {len(new_pop)}, Target: {self.size}. Adding random genome.")
            new_pop.append(Genome(self, self.genome_shape, config=self.config))

        self.members = new_pop[:self.size]

    def gather_population(self):
        all_members = []
        for species in self.species:
            all_members.extend(species.members)
        return all_members

    def get_top_genome(self):
        top_genome = sorted(self.gather_population(), key=lambda g: g.fitness, reverse=True)[0]
        return top_genome

    def save_top_genome(self, filename):
        save_path = self.config.save_path if self.config is not None else "./"
        genome = self.get_top_genome()
        with open(f'{save_path}{filename}.pkl', 'wb') as f:
            pkl.dump(genome, f)
        print(f"Top genome saved to {filename}.pkl")

