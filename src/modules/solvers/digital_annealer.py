#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import TypedDict
import logging

from modules.solvers.solver import Solver
from modules.core import Core
from utils import start_time_measurement, end_time_measurement
from dadk.BinPol import *


class DigitalAnnealer(Solver):
    """
    Class for both quantum and simulated annealing.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["Digital Annealer CPU", "Digital Annealer V3c"]

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: The name of the submodule
        :return: Instance of the default submodule
        """
        if option == "Digital Annealer CPU":
            from modules.devices.digital_annealer_cpu import DigitalAnnealerCPU  # pylint: disable=C0415
            return DigitalAnnealerCPU()
        elif option == "Digital Annealer V3c":
            from modules.devices.digital_annealer_V3c import DigitalAnnealerV3c  # pylint: disable=C0415
            return DigitalAnnealerV3c()
        else:
            raise NotImplementedError(f"Device Option {option}  not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver.

        :return: Dictionary of parameter options
        .. code-block:: python

            return {
                    "number_runs": {
                        "values": [10, 100],
                        "description": "How many reads do you need?"
                    }
                }, 
                {
                    "number_iterations": {
                        "values": [1000, 5000],
                        "description": "How many reads do you need?"
                    }
                }

        """
        return {
            "number_runs": {
                "values": [10, 100],
                "description": "How many reads do you need?"
            },
            "number_iterations": {
                "values": [1000, 5000],
                "description": "How many iterations do you need?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            number_of_reads: int

        """
        number_of_reads: int

    def run(self, mapped_problem: dict, device_wrapper: any, config: Config, **kwargs: dict) \
            -> tuple[dict, float, dict]:
        """
        Run the digital annealing solver.

        :param mapped_problem: Dict with the key 'Q' where its value should be the QUBO
        :param device_wrapper: Annealing device
        :param config: Annealing settings
        :param kwargs: Additional keyword arguments
        :return: Solution, the time it took to compute it and optional additional information
        """

        q = mapped_problem['Q']
        bp = BinPol()            
        for (var1, var2), coeff in q.items():
            var_list = [f"x_{var1[0]}_{var1[1]}"] if var1 == var2 else [f"x_{var1[0]}_{var1[1]}", f"x_{var2[0]}_{var2[1]}"]
            bp.add_term(var_list, coeff)
        additional_solver_information = {}
        device = device_wrapper.get_device()
        device.number_runs = config['number_runs']
        device.number_iterations = config['number_iterations']
        start = start_time_measurement()

        # if device_wrapper.device_name != "digital annealer":
        #     logging.error("Only digital annealer available at the moment!")
        #     logging.error("Please select another solver module.")
        #     logging.error("The benchmarking run terminates with exception.")
        #     raise Exception("Please refer to the logged error message.")

        response = device.minimize(bp)
        time_to_solve = end_time_measurement(start)

        # Take the result with the lowest energy:
        best_solution = response[0]
        sample = best_solution.configuration
        sample = response.lowest().first.sample
        logging.info(f'Annealing finished in {time_to_solve} ms.')

        return sample, time_to_solve, additional_solver_information
