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
    base_mutation_rate: float = 0.015  # Starting mutation rate
    crossover_rate: float = 0.80
    generations: int = 20000
    target_energy: float = 0.0
    seed: int = 1337

    # SIMULATED ANNEALING
    initial_temperature: float = 3.0
    final_temperature: float = 0.0001
    cooling_alpha: float = 0.9975

    # EXTINCTION EVENT
    stagnation_limit: int = 50
    catastrophic_flip_fraction: float = 0.60  # Aggressive shake-up

    # RESUME & ACTION TIME BUDGET
    resume_from_disk: bool = True
    max_runtime_seconds: int = 1200  # 20 minutes for GitHub Actions

def calculate_energy(population):
    """
    Uses vectorized NumPy FFT to find the Auto-Correlation Function (PAF)
    and calculate the energy of the entire population at once.
    """
    # population shape: (pop_size, 4, m)
    # Compute FFT along the last axis (the sequences)
    fft_vals = np.fft.fft(population, axis=-1)
    paf = np.real(np.fft.ifft(np.abs(fft_vals)**2, axis=-1))
    
    # Sum the PAF values across all 4 sequences
    total_paf = np.sum(paf, axis=1)
    
    # The objective is to make total_paf equal to 4 at all points except index 0
    # Energy is the sum of squared deviations from this target
    target = 4.0
    errors = total_paf[:, 1:] - target
    energy = np.sum(errors**2, axis=-1)
    
    # Calculate max deviation for tracking progress
    max_dev = np.max(np.abs(errors), axis=-1)
    return energy, max_dev

def run_genetic_algorithm():
    start_time = time.time()
    config = Config()
    np.random.seed(config.seed)

    # 1. LOAD PREVIOUS PROGRESS OR INITIALIZE NEW POPULATION
    file_name = "best_sequences.npy"
    if config.resume_from_disk and os.path.exists(file_name):
        print(f"Loading existing checkpoint: {file_name}")
        best_matrix = np.load(file_name)
        # Fill population by slightly mutating the best found matrix
        population = np.tile(best_matrix, (config.population_size, 1, 1))
        mask = np.random.rand(*population.shape) < 0.05
        population[mask] *= -1
        population[0] = best_matrix  # Keep exact clone as index 0
    else:
        print("No checkpoint found. Initializing random population...")
        population = np.random.choice([-1, 1], size=(config.population_size, 4, config.m)).astype(np.int8)

    energies, max_deviations = calculate_energy(population)
    best_idx = np.argmin(energies)
    global_best_energy = energies[best_idx]
    global_best_matrix = population[best_idx].copy()

    stagnation_counter = 0
    temperature = config.initial_temperature
    num_elites = int(config.population_size * config.elite_fraction)

    # 2. MAIN EVOLUTION LOOP
    for gen in range(config.generations):
        # Safety Check: Stop if running out of GitHub Actions time budget
        if time.time() - start_time > config.max_runtime_seconds:
            print(f"Time budget of {config.max_runtime_seconds}s reached. Stopping loop.")
            break

        # Sort population by lowest energy (best fitness)
        sort_indices = np.argsort(energies)
        population = population[sort_indices]
        energies = energies[sort_indices]
        max_deviations = max_deviations[sort_indices]

        # Track global improvements
        if energies[0] < global_best_energy:
            global_best_energy = energies[0]
            global_best_matrix = population[0].copy()
            stagnation_counter = 0
            # Automatically save any new discovery straight to disk
            np.save(file_name, global_best_matrix)
        else:
            stagnation_counter += 1

        # Print status updates every 10 generations
        if gen % 10 == 0:
            print(f"[GEN {gen:06d}] E={energies[0]:.6f} maxdev={max_deviations[0]:.6f} T={temperature:.6f} mut={config.base_mutation_rate * (1.0 + (stagnation_counter / 10.0)):.8f} stagnation={stagnation_counter}")

        # Target found check
        if global_best_energy <= config.target_energy:
            print(f"SUCCESS! Target energy reached at Generation {gen}!")
            break

        # EXTINCTION EVENT (If stuck for too long)
        if stagnation_counter >= config.stagnation_limit:
            print("============================================================")
            print("EXTINCTION EVENT TRIGGERED - SHAKING UP POPULATION")
            print("============================================================")
            # Keep elite, heavily scramble the rest of the population
            scramble_mask = np.random.rand(*population[num_elites:].shape) < config.catastrophic_flip_fraction
            population[num_elites:][scramble_mask] *= -1
            stagnation_counter = 0
            temperature = config.initial_temperature  # Reset annealing heat

        # AUTOMATED DYNAMIC MUTATION RATE CALCULATIONS
        current_mutation_rate = config.base_mutation_rate
        if stagnation_counter > 0:
            # Scale mutation rate up dynamically based on how long it's been stuck
            scale_factor = 1.0 + (stagnation_counter / 10.0)
            current_mutation_rate = config.base_mutation_rate * scale_factor
        
        # Keep mutation capped so it doesn't break into pure random chaos
        if current_mutation_rate > 0.05:
            current_mutation_rate = 0.05

        # 3. BREEDING NEW GENERATION (Vectorized Operations)
        new_population = np.empty_index = np.empty_like(population)
        new_population[:num_elites] = population[:num_elites]  # Elitism

        # Selection using simulated annealing probabilities
        weights = np.exp(-energies / temperature)
        probabilities = weights / np.sum(weights)
        
        parent1_idx = np.random.choice(config.population_size, size=config.population_size - num_elites, p=probabilities)
        parent2_idx = np.random.choice(config.population_size, size=config.population_size - num_elites, p=probabilities)
        
        parents1 = population[parent1_idx]
        parents2 = population[parent2_idx]

        # Vectorized Ring Crossover
        crossover_mask = np.random.rand(config.population_size - num_elites, 1, 1) < config.crossover_rate
        cutoff_points = np.random.randint(0, config.m, size=(config.population_size - num_elites, 1, 1))
        idx_matrix = np.arange(config.m).reshape(1, 1, config.m)
        
        left_side = idx_matrix < cutoff_points
        crossover_filter = crossover_mask & left_side
        
        children = np.where(crossover_filter, parents1, parents2)

        # Dynamic Vectorized Mutation
        mutation_mask = np.random.rand(*children.shape) < current_mutation_rate
        children[mutation_mask] *= -1

        new_population[num_elites:] = children
        population = new_population

        # Recalculate energy states for next loop iteration
        energies, max_deviations = calculate_energy(population)
        temperature = max(config.final_temperature, temperature * config.cooling_alpha)

    # 4. FINAL CLEANUP AND EXPORT
    print(f"Search batch completed in {time.time() - start_time:.2f} sec")
    print(f"Current Best Energy = {global_best_energy:.6f}")
    np.save(file_name, global_best_matrix)
    print("Progress safely saved to best_sequences.npy")

if __name__ == "__main__":
    run_genetic_algorithm()    max_runtime_seconds: int = 1200  # Set slightly under 30 mins to finish and cache safely


CFG = Config()

random.seed(CFG.seed)
np.random.seed(CFG.seed)

# ============================================================
# OPTIMIZED VECTORIZED OPERATIONS
# ============================================================

def random_population():
    return np.random.choice(
        [-1, 1],
        size=(CFG.population_size, 4, CFG.m)
    ).astype(np.int8)


def compute_population_metrics(population):
    """Calculates both Energy and Maxdev in a single FFT pass."""
    F = np.fft.fft(population, axis=-1)
    PSD = np.abs(F) ** 2
    spectral_sum = np.sum(PSD, axis=1)
    target = 4 * CFG.m
    defect = spectral_sum - target
    
    energies = np.sum(defect ** 2, axis=1)
    maxdev = np.max(np.abs(defect), axis=1)
    return energies, maxdev


def ring_crossover_vectorized(p1_mates, p2_mates):
    """Performs Ring Crossover on the entire array without loops."""
    count = len(p1_mates)
    children = p1_mates.copy()
    
    # Generate random cut points for all individuals at once
    cut1 = np.random.randint(0, CFG.m, size=(count, 4, 1))
    cut2 = np.random.randint(0, CFG.m, size=(count, 4, 1))
    
    start = np.minimum(cut1, cut2)
    end = np.maximum(cut1, cut2)
    
    # Create an index grid to mask the interval ranges
    grid = np.arange(CFG.m).reshape(1, 1, CFG.m)
    mask = (grid >= start) & (grid < end)
    
    children[mask] = p2_mates[mask]
    return children


def mutate_population_vectorized(population, mutation_rate):
    """Flips matrix bits across the whole population instantly."""
    total_bits = 4 * CFG.m
    flips_per_individual = max(1, int(mutation_rate * total_bits))
    P = population.shape[0]
    
    for _ in range(flips_per_individual):
        seq_idx = np.random.randint(0, 4, size=P)
        bit_idx = np.random.randint(0, CFG.m, size=P)
        population[np.arange(P), seq_idx, bit_idx] *= -1
    return population


def catastrophic_mutation_vectorized(population, best):
    """Triggers an extinction event instantly across the array."""
    print("\n" + "=" * 60)
    print("EXTINCTION EVENT TRIGGERED")
    print("=" * 60)

    new_population = population.copy()
    total_bits = 4 * CFG.m
    flips = int(CFG.catastrophic_flip_fraction * total_bits)
    P = len(new_population)

    for i in range(1, P):
        seq_idx = np.random.randint(0, 4, size=flips)
        bit_idx = np.random.randint(0, CFG.m, size=flips)
        new_population[i, seq_idx, bit_idx] *= -1

    new_population[0] = best.copy()
    return new_population


def load_baseline():
    if CFG.resume_from_disk and os.path.exists("best_sequences.npy"):
        print("\nLoading baseline from disk...\n")
        return np.load("best_sequences.npy").astype(np.int8)
    return None


def save_best(best):
    np.save("best_sequences.npy", best.astype(np.int8))


class TemperatureSchedule:
    def __init__(self):
        self.temperature = CFG.initial_temperature

    def update(self):
        self.temperature = max(self.temperature * CFG.cooling_alpha, CFG.final_temperature)

    def mutation_rate(self):
        return CFG.base_mutation_rate * self.temperature


# ============================================================
# EVOLUTION ENGINE
# ============================================================

class WilliamsonGA:
    def __init__(self):
        self.population = random_population()
        baseline = load_baseline()

        if baseline is not None:
            self.population[0] = baseline
            for i in range(1, 8):
                clone = baseline.copy()
                noise = np.random.randint(0, CFG.m, size=8)
                for k in range(4):
                    clone[k, noise] *= -1
                self.population[i] = clone

        self.best = self.population[0].copy()
        self.best_energy = float("inf")
        self.stagnation = 0
        self.temperature = TemperatureSchedule()

    def evolve(self):
        start_time = time.time()

        for gen in range(CFG.generations):
            if time.time() - start_time > CFG.max_runtime_seconds:
                print("\n[TIME LIMIT] Stopping early to save progress on GitHub!")
                break

            # 1. Single Metrics Pass
            energies, maxdevs = compute_population_metrics(self.population)
            fitness = -energies - 100.0 * maxdevs
            
            order = np.argsort(fitness)[::-1]
            self.population = self.population[order]
            energies = energies[order]
            maxdevs = maxdevs[order]

            best = self.population[0]
            E = energies[0]
            D = maxdevs[0]

            if E < self.best_energy:
                self.best_energy = E
                self.best = best.copy()
                self.stagnation = 0
                save_best(self.best)
            else:
                self.stagnation += 1

            mutation_rate = self.temperature.mutation_rate()

            # Reduce logging slightly to speed up string standard output handling
            if gen % 20 == 0 or E <= CFG.target_energy:
                print(f"[GEN {gen:06d}] E={E:.4f} maxdev={D:.4f} T={self.temperature.temperature:.4f} stagnation={self.stagnation}")

            if E <= CFG.target_energy:
                print("\nTARGET ENERGY ACHIEVED\n")
                save_best(best)
                return best

            if self.stagnation >= CFG.stagnation_limit:
                self.population = catastrophic_mutation_vectorized(self.population, self.best)
                self.stagnation = 0

            # 2. Elite Isolation
            elite_count = max(2, int(CFG.elite_fraction * CFG.population_size))
            elites = self.population[:elite_count]

            # 3. Vectorized Breeding Block
            children_needed = CFG.population_size - elite_count
            
            p1_idx = np.random.randint(0, elite_count, size=children_needed)
            p2_idx = np.random.randint(0, elite_count, size=children_needed)
            
            p1_mates = elites[p1_idx]
            p2_mates = elites[p2_idx]
            
            children = ring_crossover_vectorized(p1_mates, p2_mates)
            
            # Revert to parent 1 if crossover probability checks out false
            no_cross_mask = np.random.rand(children_needed) >= CFG.crossover_rate
            children[no_cross_mask] = p1_mates[no_cross_mask]

            # 4. Mutations and Reassembly
            children = mutate_population_vectorized(children, mutation_rate)
            self.population = np.concatenate([elites, children], axis=0)
            self.temperature.update()

        return self.best


def main():
    print("=" * 70)
    print("ANNEALED VECTORIZED FFT WILLIAMSON SEARCH ENGINE")
    print("=" * 70)

    start = time.time()
    engine = WilliamsonGA()
    result = engine.evolve()
    elapsed = time.time() - start

    print(f"\nSearch batch completed in {elapsed:.2f} sec")
    energies, _ = compute_population_metrics(result[np.newaxis, ...])
    print(f"Current Best Energy = {energies[0]:.6f}")
    
    save_best(result)
    print("Progress safely saved to best_sequences.npy")


if __name__ == "__main__":
    main()    max_runtime_seconds: int = 2700  # 45 minutes safety timer for GitHub Actions


CFG = Config()

random.seed(CFG.seed)
np.random.seed(CFG.seed)

# ============================================================
# OPERATIONS
# ============================================================

def random_population():
    return np.random.choice(
        [-1, 1],
        size=(CFG.population_size, 4, CFG.m)
    ).astype(np.int8)


def compute_population_energy(population):
    F = np.fft.fft(population, axis=-1)
    PSD = np.abs(F) ** 2
    spectral_sum = np.sum(PSD, axis=1)
    target = 4 * CFG.m
    defect = spectral_sum - target
    energies = np.sum(defect ** 2, axis=1)
    return energies


def compute_population_maxdev(population):
    F = np.fft.fft(population, axis=-1)
    PSD = np.abs(F) ** 2
    spectral_sum = np.sum(PSD, axis=1)
    target = 4 * CFG.m
    defect = np.abs(spectral_sum - target)
    return np.max(defect, axis=1)


def compute_fitness(population):
    E = compute_population_energy(population)
    D = compute_population_maxdev(population)
    return -E - 100.0 * D


class TemperatureSchedule:
    def __init__(self):
        self.temperature = CFG.initial_temperature

    def update(self):
        self.temperature *= CFG.cooling_alpha
        self.temperature = max(self.temperature, CFG.final_temperature)

    def mutation_rate(self):
        return CFG.base_mutation_rate * self.temperature


def ring_crossover_sequence(a, b):
    n = len(a)
    p1 = np.random.randint(0, n)
    p2 = np.random.randint(0, n)
    if p1 > p2:
        p1, p2 = p2, p1

    child = a.copy()
    if p1 < p2:
        child[p1:p2] = b[p1:p2]
    else:
        child[p1:] = b[p1:]
        child[:p2] = b[:p2]
    return child


def ring_crossover(parent1, parent2):
    child = np.empty_like(parent1)
    for k in range(4):
        child[k] = ring_crossover_sequence(parent1[k], parent2[k])
    return child


def mutate_population(population, mutation_rate):
    P = population.shape[0]
    total_bits = 4 * CFG.m
    flips_per_individual = max(1, int(mutation_rate * total_bits))

    for i in range(P):
        seq_idx = np.random.randint(0, 4, size=flips_per_individual)
        bit_idx = np.random.randint(0, CFG.m, size=flips_per_individual)
        population[i, seq_idx, bit_idx] *= -1
    return population


def catastrophic_mutation(population, best):
    print("\n" + "=" * 60)
    print("EXTINCTION EVENT TRIGGERED")
    print("=" * 60)

    new_population = population.copy()
    total_bits = 4 * CFG.m
    flips = int(CFG.catastrophic_flip_fraction * total_bits)

    for i in range(1, len(new_population)):
        seq_idx = np.random.randint(0, 4, size=flips)
        bit_idx = np.random.randint(0, CFG.m, size=flips)
        new_population[i, seq_idx, bit_idx] *= -1

    new_population[0] = best.copy()
    return new_population


def load_baseline():
    if CFG.resume_from_disk and os.path.exists("best_sequences.npy"):
        print("\nLoading baseline from disk...\n")
        return np.load("best_sequences.npy").astype(np.int8)
    return None


def save_best(best):
    np.save("best_sequences.npy", best.astype(np.int8))


# ============================================================
# EVOLUTION ENGINE
# ============================================================

class WilliamsonGA:
    def __init__(self):
        self.population = random_population()
        baseline = load_baseline()

        if baseline is not None:
            self.population[0] = baseline
            for i in range(1, 8):
                clone = baseline.copy()
                noise = np.random.randint(0, CFG.m, size=8)
                for k in range(4):
                    clone[k, noise] *= -1
                self.population[i] = clone

        self.best = self.population[0].copy()
        self.best_energy = float("inf")
        self.stagnation = 0
        self.temperature = TemperatureSchedule()

    def evolve(self):
        start_time = time.time()

        for gen in range(CFG.generations):
            # Check if GitHub Action is running out of time
            if time.time() - start_time > CFG.max_runtime_seconds:
                print("\n[TIME LIMIT] Stopping early to save progress on GitHub!")
                break

            fitness = compute_fitness(self.population)
            order = np.argsort(fitness)[::-1]
            self.population = self.population[order]
            fitness = fitness[order]

            best = self.population[0]
            E = compute_population_energy(best[np.newaxis, ...])[0]
            D = compute_population_maxdev(best[np.newaxis, ...])[0]

            if E < self.best_energy:
                self.best_energy = E
                self.best = best.copy()
                self.stagnation = 0
                save_best(self.best)
            else:
                self.stagnation += 1

            mutation_rate = self.temperature.mutation_rate()

            if gen % 10 == 0 or E <= CFG.target_energy:
                print(
                    f"[GEN {gen:06d}] "
                    f"E={E:.6f} "
                    f"maxdev={D:.6f} "
                    f"T={self.temperature.temperature:.6f} "
                    f"mut={mutation_rate:.8f} "
                    f"stagnation={self.stagnation}"
                )

            if E <= CFG.target_energy:
                print("\nTARGET ENERGY ACHIEVED\n")
                save_best(best)
                return best

            if self.stagnation >= CFG.stagnation_limit:
                self.population = catastrophic_mutation(self.population, self.best)
                self.stagnation = 0

            elite_count = max(2, int(CFG.elite_fraction * CFG.population_size))
            elites = self.population[:elite_count]

            children = []
            while len(children) < (CFG.population_size - elite_count):
                p1 = elites[np.random.randint(0, elite_count)]
                p2 = elites[np.random.randint(0, elite_count)]

                if np.random.rand() < CFG.crossover_rate:
                    child = ring_crossover(p1, p2)
                else:
                    child = p1.copy()
                children.append(child)

            children = np.array(children, dtype=np.int8)
            children = mutate_population(children, mutation_rate)
            self.population = np.concatenate([elites, children], axis=0)
            self.temperature.update()

        return self.best


def main():
    print("=" * 70)
    print("ANNEALED FFT WILLIAMSON SEARCH ENGINE (WORKFLOW OPTIMIZED)")
    print("=" * 70)

    start = time.time()
    engine = WilliamsonGA()
    result = engine.evolve()
    elapsed = time.time() - start

    print(f"\nSearch batch completed in {elapsed:.2f} sec")
    final_energy = compute_population_energy(result[np.newaxis, ...])[0]
    print(f"Current Best Energy = {final_energy:.6f}")
    
    save_best(result)
    print("Progress safely saved to best_sequences.npy")


if __name__ == "__main__":
    main()
