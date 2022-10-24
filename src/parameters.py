class Parameters:
    def __init__(self, json_path: str = "") -> None:
        """Model's parameters:
        
        generations (int): number of generations
        population_size (int): size of the population
        elitism_size (int): number of the best fit chromosomes
            which will go to the next generation without
            the tournament
        mating_prob (float): probability with which chromosome 
            added to mating pool
        crossover_prob (float): probability with which two chromosomes
            go to the next generation themselves, 
            otherwise they generate two child
        mutation_prob (float): genes mutation probability
        track_history (bool): flag indicating whether 
            to track the history or not

        Args:
            json_path (str, optional): path to the json file with parameters.
            Defaults to "".
        """
        self.generations = 10
        self.population_size = 10
        self.elitism = 1
        self.mating_prob = 0.85
        self.crossover_prob = 0.75
        self.mutation_prob = 0.1
        self.track_history = False
        self.kb = 3
        self.ke = 3
        
        if json_path:
            self._read_json()
            
    def _read_json(self) -> None:
        # TODO
        pass
