from typing import List, Optional, Union

import numpy as np
import sk2torch
import torch
import json
from guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction as GuacamolScoringFunction
from mso.objectives.scoring import ScoringFunction as MSOScoringFunction
from mso.optimizer import BasePSOptimizer
from rdkit import Chem
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

from models.global_utils import smiles_from_file, SMILES_DIR
from models.inference import InferenceBase



class RandomGenerator(DistributionMatchingGenerator):
    def __init__(self, inference_server: InferenceBase):
        self.inference_server = inference_server

    def generate(self, number_samples: int):
        return self.inference_server.sample_molecules(number_samples)
    

class GoalDirectedWrapper(GoalDirectedGenerator):
    def __init__(
        self,
        inference_server: InferenceBase,
        opt_config: dict,
        dataset: str,
        benchmark: str,
        optimization_method: str,
    ):
        self.inference_server = inference_server
        self.opt_config = opt_config
        self.dataset = dataset
        self.benchmark = benchmark
        self.method = optimization_method
        if self.method == "mso":
            self.estimate_radius()
        elif self.method == "gasc":
            self.train_proxy_oracle()
        else:
            raise NotImplementedError()

    def get_training_smiles(self):
        smiles = smiles_from_file(SMILES_DIR / self.dataset / "train.txt")
        smiles = smiles[: self.opt_config["num_train_samples"]]
        return smiles

    def train_proxy_oracle(self):
        benchmark = goal_directed_benchmark_suite(version_name=self.benchmark)
        objectives = [b.wrapped_objective for b in benchmark]
        self.objectives = [o.name for o in objectives]
        train_smiles = self.get_training_smiles()
        target_values = np.array([o.score_list(train_smiles) for o in objectives]).T
        embeddings = self.inference_server.encode_molecules(train_smiles)

        sklearn_proxy = MLPRegressor()
        sklearn_proxy.fit(embeddings, target_values)
        torch_proxy = sk2torch.wrap(sklearn_proxy)
        torch_proxy.cuda()
        torch_proxy.eval()
        self.proxy_oracle = torch_proxy

    def wrap_scoring_function(self, scoring_function):
        if isinstance(scoring_function, GuacamolScoringFunction):
            return MSOScoringFunction(
                lambda x: scoring_function.score(Chem.MolToSmiles(x)),
                name=scoring_function.name,
                is_mol_func=True,
            )
        else:
            return scoring_function

    def estimate_radius(self):
        print("Estimating latent radius...")
        training_smiles = self.get_training_smiles()
        latent_vectors = self.inference_server.encode_molecules(training_smiles)
        latent_radius = np.max(np.linalg.norm(latent_vectors, axis=1))
        self.latent_radius = latent_radius
        print("Estimated latent radius: ", latent_radius)

    def generate_optimized_molecules(
        self,
        scoring_function: Union[GuacamolScoringFunction, MSOScoringFunction],
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:
        scoring_function_wrapped = self.wrap_scoring_function(scoring_function)
        if self.method == "mso":
            optimization_method = self.optimize_molecules_with_mso
        elif self.method == "gasc":
            optimization_method = self.optimize_molecules_with_gasc
        else:
            raise NotImplementedError
        return optimization_method(
            scoring_function=scoring_function_wrapped,
            number_molecules=number_molecules,
            starting_population=starting_population,
        )

    def optimize_molecules_with_gasc(
        self,
        scoring_function: MSOScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
        disable_progress_bar: bool = False,
    ) -> List[str]:
        """
        Perform gradient ascent optimization on a given scoring function
        """
        assert scoring_function.name in self.objectives
        if starting_population is None:
            starting_population = self.inference_server.sample_molecules(number_molecules)

        latent_vectors_current = self.inference_server.encode_molecules(starting_population)
        latent_vectors_current = torch.tensor(latent_vectors_current).cuda()
        latent_vectors_current.requires_grad = True
        decoding_trajectory = []
        optimizer = torch.optim.Adam([latent_vectors_current], lr=self.opt_config["gasc"]["lr"])
        for _ in tqdm(range(self.opt_config["gasc"]["num_steps"]), disable=disable_progress_bar):
            scores = self.proxy_oracle(latent_vectors_current)
            scores = scores[:, self.objectives.index(scoring_function.name)]
            loss = -scores.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            decoded_mols = self.inference_server.decode(latent_vectors_current.detach().cpu().numpy())
            decoding_trajectory.append(decoded_mols)
        decoding_trajectory = np.array(decoding_trajectory).T

        score_lambda = lambda x: scoring_function([Chem.MolFromSmiles(x)])[0].item()
        extract_best_lambda = lambda x: max([t for t in x], key=lambda y: score_lambda(y))
        optimized_molecules = [str(extract_best_lambda(x)) for x in decoding_trajectory]
        return optimized_molecules

    def optimize_molecules_with_mso(
        self,
        scoring_function: MSOScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
        disable_progress_bar: bool = False,
    ) -> List[str]:
        """
        Perform MSO optimization on a given scoring function
        """
        optimized_molecules = []
        with tqdm(total=number_molecules, desc=scoring_function.name, disable=disable_progress_bar) as progress_bar:
            while len(optimized_molecules) < number_molecules:
                if starting_population is None:
                    starting_population = self.inference_server.sample_molecules(
                        self.opt_config["mso"]["num_starting_mols"]
                    )
                opt = BasePSOptimizer.from_query(
                    starting_population,
                    num_part=self.opt_config["mso"]["num_particles"],
                    num_swarms=self.opt_config["mso"]["num_swarms"],
                    inference_model=self.inference_server,
                    scoring_functions=[scoring_function],
                    x_max=self.latent_radius,
                )
                opt.run(self.opt_config["mso"]["num_runs"], verbose=False)
                opt_history = opt.best_fitness_history.reset_index(drop=True)

                optimized_molecule = opt_history.iloc[-1]["smiles"]
                if optimized_molecule not in optimized_molecules:
                    optimized_molecules.append(optimized_molecule.item())
                    progress_bar.update(1)
        return optimized_molecules
    

def read_results_from_file(json_path):
    out_results = dict()
    with open(json_path) as json_file:
        guacamol_results = json.load(json_file)
        for subdict in guacamol_results["results"]:
            out_results[subdict["benchmark_name"]] = subdict["score"]
    return out_results
