import numpy as np
from random import uniform
from typing import Any, Callable

from src.ems import EMS
from src.cargo import Cargo, CargoID
from src.chromosome import Chromosome
from src.parameters import Parameters
from src.container import Container, ContainerID
from src.placement import Placement, PlacementIndexed
from src.utils import (
    argsort_by_func, construct_encoder_decoder, 
    different_rand_vals, smallest_coord_elem_index
)


class Population:
    """Population class"""

    def __init__(
        self, 
        cargos_source: list[Cargo], containers_source: list[Container],
        params: Parameters
    ) -> None:
        """Initialize population class

        Args:
            cargos_source (list): cargos to pack
            containers_source (list): containers to use for package
            params (Parameters): model's parameters
        """
        cargos_encoder, cargos_decoder = construct_encoder_decoder(cargos_source)
        self.cargos_encoder: dict[Cargo, CargoID] = cargos_encoder
        self.cargos_decoder: dict[CargoID, Cargo] = cargos_decoder
        
        containers_encoder, containers_decoder = construct_encoder_decoder(containers_source)
        self.containers_encoder: dict[Container, ContainerID] = containers_encoder
        self.containers_decoder: dict[ContainerID, Container] = containers_decoder

        self.params = params

        self.population = self._init_population()
        self.fitness = self._init_fitness()
        self.history = self._init_history(self.params.track_history)
        
        self._next = self._get_next_function(self.params.track_history)

    def _init_population(self) -> np.ndarray[Any, Chromosome]:
        """Initialize random population

        Returns:
            np.ndarray[Any, Chromosome]: population
        """
        cargos = list(self.cargos_decoder.keys())
        containers = list(self.containers_decoder.keys())

        population = np.empty(self.params.population_size, Chromosome)
        bps = np.array(cargos)
        
        def get_dim(c: int, i: int) -> int:
            return self.cargos_decoder[c].dims[i]
        
        vect_get_dim = np.vectorize(get_dim)

        for i in range(4):
            np.random.shuffle(containers)
            population[i] = Chromosome(
                argsort_by_func(bps, lambda c: vect_get_dim(c, i)),
                containers
            )

        for i in range(4, self.params.population_size):
            np.random.shuffle(bps)
            np.random.shuffle(containers)
            population[i] = Chromosome(bps, containers)
            
        return population

    def _init_fitness(self) -> np.ndarray[Any, Chromosome]:
        """Initialize fitness tracker

        Returns:
            np.ndarray(Any, int): fitness tracker
        """
        return np.empty(self.params.population_size, int)

    @staticmethod
    def _init_history(
        track_history: bool
    ) -> list[tuple[list[Chromosome], list[int]]] | None:
        """Initialize history tracker

        Args:
            track_history (bool): flag indicating whether 
            to track the history or not

        Returns:
            list[tuple[list[Chromosome], list[int]]] | None: history tracker
        """
        if track_history:
            return []
        return None

    def _get_next_function(self, track_history: bool) -> Callable[[], None]:
        """Get next generation function

        Args:
            track_history (bool): flag indicating whether 
            to track the history or not

        Returns:
            Callable[[], None]: next generation function
        """
        if track_history:
            return self._next_with_tracking
        else:
            return self._next_without_tracking

    def _next_with_tracking(self) -> None:
        """Generate next population with history tracking
        """
        self.history.append((self.population.tolist(), self.fitness.tolist()))
        self._next_without_tracking()

    def _next_without_tracking(self) -> None:
        """Generate next population without history tracking
        """
        # Evaluate current fitness
        self._evaluate_fitness()

        # Get current generation sorted by its fitness
        not_elite = self.params.population_size - self.params.elitism
        sorted_indeces = np.argpartition(self.fitness, not_elite - 1)
        pop_sorted = self.population.copy()[sorted_indeces]
        fit_sorted = self.fitness.copy()[sorted_indeces]

        # Elitism
        self.population[not_elite:] = pop_sorted[not_elite:]
        pop_sorted = pop_sorted[:not_elite]
        fit_sorted = fit_sorted[:not_elite]

        # Construct mating pool
        mating_pool = np.empty(not_elite, Chromosome)

        for i in range(not_elite):
            r1, r2 = different_rand_vals(0, not_elite - 1)
            f1, f2 = fit_sorted[r1], fit_sorted[r2]

            if f1 > f2:
                min_c, max_c = r2, r1
            else:
                min_c, max_c = r1, r2

            if uniform(0, 1) > self.params.mating_prob:
                mating_pool[i] = pop_sorted[min_c]
            else:
                mating_pool[i] = pop_sorted[max_c]

        # Crossover and mutation
        for i in range(0, not_elite - 1, 2):
            c1, c2 = mating_pool[i:i+2]

            if uniform(0, 1) > self.params.crossover_prob:
                o1, o2 = c1, c2
            else:
                o1, o2 = self._mate(c1, c2)
                self._mutate((o1, o2))

            self.population[i], self.population[i + 1] = o1, o2

    @staticmethod
    def _mate(
        parent1: Chromosome, parent2: Chromosome
    ) -> tuple[Chromosome, Chromosome]:
        """Mate two chromosomes to get two child chromosomes from them

        Args:
            parent1 (Chromosome): parent chromosome 1
            parent2 (Chromosome): parent chromosome 2

        Returns:
            tuple[Chromosome, Chromosome]: child chromosomes
        """
        return parent1.mate(parent2), parent2.mate(parent1)
    
    def _mutate(self, chromosomes: tuple[Chromosome, Chromosome]) -> None:
        """Mutate the given chromosomes

        Args:
            chromosomes (tuple[Chromosome, Chromosome]): chromosomes to mutate
        """
        for c in chromosomes:
            if uniform(0, 1) > self.params.mutation_prob:
                c.mutate()

    def _evaluate_fitness(self) -> None:
        """Evaluate fitness for the current population
        """
        for c in range(self.params.population_size):
            self.fitness[c] = self._best_match_heuristic(self.population[c])

    def _best_match_heuristic(self, chromosome: Chromosome) -> int:
        bps: list[CargoID] = chromosome.bps.tolist()
        
        made_placements: list[Placement] = []
        
        opened_containers = np.array([], dtype=int)
        container_spaces: dict[ContainerID, list[EMS]] = {self.con}
        
        while len(bps) > 0:
            candidate_placements: list[PlacementIndexed] = []
            box_placed = False
            
            for container in opened_containers:
                spaces = container_spaces[container]
                j = 0
                
                while j < len(spaces[container]) and not box_placed:                
                    k = j + self.params.ke
                    
                    while j < k and j < len(spaces[container]):
                        for i in range(self.params.kb):
                            if i >= len(bps):
                                break
                            
                            cargo = self.cargos_decoder[bps[i]]
                            valid_placements = cargo.valid_placements(
                                spaces[container][j]
                            )
                            valid_placements_indexed = (
                                PlacementIndexed(i, placement)
                                for placement in valid_placements
                            )
                            
                            candidate_placements = np.append(
                                candidate_placements, 
                                valid_placements_indexed
                            )
                        
                        j += 1
                        
                    if len(candidate_placements) > 0:
                        ind = smallest_coord_elem_index(candidate_placements)
                        bps.pop(candidate_placements[ind].index)
                        made_placements.append(
                            candidate_placements[ind].placement
                        )
                        
                        # TODO: update EMS
                        
                        box_placed = True
                        
                if box_placed:
                    break
                            
        # TODO: lines 23+ from article
        
        fitness: int = 0
        return fitness