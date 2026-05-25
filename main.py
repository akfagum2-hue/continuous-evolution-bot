from __future__ import annotations

import os
import time
import random
import numpy as np

from dataclasses import dataclass

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    m: int = 167
    n: int = 668
    population_size: int = 256
    elite_fraction: float = 0.10
    base_mutation_rate: float = 0.003
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
    catastrophic_flip_fraction: float = 0.30

    # RESUME & ACTION TIME BUDGET
    resume_from_disk: bool = True
    max_runtime_seconds: int = 2700  # 45 minutes safety timer for GitHub Actions


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
