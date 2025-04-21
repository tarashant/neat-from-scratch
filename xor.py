from neat import *

XOR_TABLE = [
    (0,0,0),
    (0,1,1),
    (1,0,1),
    (1,1,0),
]

def evaluate_genome(genome):
    error = 0
    for x1, x2, target in XOR_TABLE:
        output = genome.forward([x1, x2, 1.0])
        error += abs(target - output[0])
    genome.fitness = 4.0 - error
    return genome.fitness

def run_xor_experiment(num_gens=100, pop_size=100):
    config = NEATConfig(
        genome_shape=(3, 1),
        population_size=pop_size,
        add_node_mutation_prob=0.05,
        add_conn_mutation_prob=0.08,
        sigma=0.1,
        perturb_prob=0.8,
        reset_prob=0.1,
        species_threshold=3.0,
        num_elites=1,
        selection_share=0.2,
    )
    print(config)
    population = Population(config=config)
    
    fitnesses = []
    avg_fitness = 0
    top_fitness = 0

    for gen in range(num_gens):
        for genome in population.members:
            fitness = evaluate_genome(genome)
            fitnesses.append(fitness)

        top_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)

        print(f"Gen {gen+1} | Average fitness: {avg_fitness} | Top fitness: {top_fitness} ")

        population.reproduce()

        population.species = []
        for genome in population.members:
            population.speciate(genome)

    top_genome = population.get_top_genome()
    return top_genome, avg_fitness, top_fitness

if __name__ == '__main__':
    top_genome, avg_fitness, top_fitness = run_xor_experiment(num_gens=100, pop_size=500)
    print("\n=== Testing Best Genome ===")
    for x1, x2, target in XOR_TABLE:
        output = top_genome.forward([x1, x2, 1.0])[0]
        print(f" Input: {[x1, x2]} → Predicted: {output:.3f}   (target {target})")