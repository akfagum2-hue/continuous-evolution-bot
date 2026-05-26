import os
import time
from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    m: int = 167
    n: int = 668  # 4 sequences * 167
    population_size: int = 256
    elite_fraction: float = 0.10
    base_mutation_rate: float = 0.015  
    crossover_rate: float = 0.80
    generations: int = 50000  # Upgraded to run longer per batch
    target_energy: float = 0.0
    seed: int = 1337

    # SIMULATED ANNEALING
    initial_temperature: float = 3.0
    final_temperature: float = 0.0001
    cooling_alpha: float = 0.9975

    # EXTINCTION EVENT
    stagnation_limit: int = 50
    catastrophic_flip_fraction: float = 0.60  

    # RESUME & ACTION TIME BUDGET
    resume_from_disk: bool = True
    max_runtime_seconds: int = 1200  # 20 minutes safe buffer

def calculate_energy(population):
    """Calculates both Energy and Maxdev in a single FFT pass."""
    fft_vals = np.fft.fft(population, axis=-1)
    paf = np.real(np.fft.ifft(np.abs(fft_vals)**2, axis=-1))
    total_paf = np.sum(paf, axis=1)
    
    target = 4.0
    errors = total_paf[:, 1:] - target
    energy = np.sum(errors**2, axis=-1)
    max_dev = np.max(np.abs(errors), axis=-1)
    return energy, max_dev

def run_genetic_algorithm():
    start_time = time.time()
    config = Config()
    np.random.seed(config.seed)

    # 1. LOAD PREVIOUS PROGRESS OR INITIALIZE
    file_name = "best_sequences.npy"
    if config.resume_from_disk and os.path.exists(file_name):
        print(f"Loading existing checkpoint: {file_name}")
        best_matrix = np.load(file_name).astype(np.int8)
        population = np.tile(best_matrix, (config.population_size, 1, 1))
        mask = np.random.rand(*population.shape) < 0.05
        population[mask] *= -1
        population[0] = best_matrix  
    else:
        print("No checkpoint found. Initializing random population...")
        population = np.random.choice([-1, 1], size=(config.population_size, 4, config.m)).astype(np.int8)

    energies, max_deviations = calculate_energy(population)
    best_idx = np.argmin(energies)
    global_best_energy = energies[best_idx]
    global_best_matrix = population[best_idx].copy()

    stagnation_counter = 0
    temperature = config.initial_temperature
    num_elites = max(2, int(config.population_size * config.elite_fraction))

    # 2. MAIN EVOLUTION LOOP
    for gen in range(config.generations):
        if time.time() - start_time > config.max_runtime_seconds:
            print(f"\n[TIME LIMIT] Stopping loop at {config.max_runtime_seconds}s to save progress!")
            break

        sort_indices = np.argsort(energies)
        population = population[sort_indices]
        energies = energies[sort_indices]
        max_deviations = max_deviations[sort_indices]

        if energies[0] < global_best_energy:
            global_best_energy = energies[0]
            global_best_matrix = population[0].copy()
            stagnation_counter = 0
            np.save(file_name, global_best_matrix)
        else:
            stagnation_counter += 1

        # AUTOMATED DYNAMIC MUTATION RATE
        current_mutation_rate = config.base_mutation_rate
        if stagnation_counter > 0:
            scale_factor = 1.0 + (stagnation_counter / 10.0)
            current_mutation_rate = config.base_mutation_rate * scale_factor
        
        if current_mutation_rate > 0.05:
            current_mutation_rate = 0.05

        if gen % 20 == 0:
            print(f"[GEN {gen:06d}] E={energies[0]:.4f} maxdev={max_deviations[0]:.4f} T={temperature:.4f} mut={current_mutation_rate:.6f} stagnation={stagnation_counter}")

        if global_best_energy <= config.target_energy:
            print(f"\nSUCCESS! Target energy reached at Generation {gen}!")
            break

        # EXTINCTION EVENT
        if stagnation_counter >= config.stagnation_limit:
            print("\n============================================================")
            print("EXTINCTION EVENT TRIGGERED - SHAKING UP POPULATION")
            print("============================================================")
            scramble_mask = np.random.rand(*population[num_elites:].shape) < config.catastrophic_flip_fraction
            population[num_elites:][scramble_mask] *= -1
            stagnation_counter = 0
            temperature = config.initial_temperature  

        # 3. BREEDING (Vectorized Operations)
        new_population = np.empty_like(population)
        new_population[:num_elites] = population[:num_elites]  

        weights = np.exp(-energies / temperature)
        probabilities = weights / np.sum(weights)
        
        children_needed = config.population_size - num_elites
        parent1_idx = np.random.choice(config.population_size, size=children_needed, p=probabilities)
        parent2_idx = np.random.choice(config.population_size, size=children_needed, p=probabilities)
        
        parents1 = population[parent1_idx]
        parents2 = population[parent2_idx]

        # Vectorized Ring Crossover
        crossover_mask = np.random.rand(children_needed, 1, 1) < config.crossover_rate
        cutoff_points = np.random.randint(0, config.m, size=(children_needed, 1, 1))
        idx_matrix = np.arange(config.m).reshape(1, 1, config.m)
        
        left_side = idx_matrix < cutoff_points
        crossover_filter = crossover_mask & left_side
        
        children = np.where(crossover_filter, parents1, parents2)

        # Dynamic Vectorized Mutation
        mutation_mask = np.random.rand(*children.shape) < current_mutation_rate
        children[mutation_mask] *= -1

        new_population[num_elites:] = children
        population = new_population

        energies, max_deviations = calculate_energy(population)
        temperature = max(config.final_temperature, temperature * config.cooling_alpha)

    # 4. FINAL CLEANUP AND EXPORT
    print(f"\nSearch batch completed in {time.time() - start_time:.2f} sec")
    print(f"Current Best Energy = {global_best_energy:.6f}")
    np.save(file_name, global_best_matrix)
    print("Progress safely saved to best_sequences.npy")

if __name__ == "__main__":
    print("=" * 70)
    print("ANNEALED VECTORIZED FFT WILLIAMSON SEARCH ENGINE")
    print("=" * 70)
    run_genetic_algorithm()
