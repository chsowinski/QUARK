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
import logging
from typing import TypedDict
from more_itertools import locate

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from modules.applications.Mapping import *
from utils import start_time_measurement, end_time_measurement


class Ising(Mapping):
    """
    Ising formulation of the auto-carrier loading (ACL) problem.

    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["QAOA", "QiskitQAOA"]
        self.global_variables = 0
        logging.warning("Currently, all scenarios are too large to be solved with an Ising model.")
        logging.warning("Consider using another mapping until the modelling is refined.")

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "numpy",
                "version": "1.23.5"
            },
            {
                "name": "more-itertools",
                "version": "9.0.0"
            },
            {
                "name": "qiskit-optimization",
                "version": "0.5.0"
            },
        ]

    def get_parameter_options(self):
        """
        Returns empty dict as this mapping has no configurable settings.

        :return: empty dict
        :rtype: dict
        """
        return {
        }

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings.
        """
        pass

    def map_pulp_to_qiskit(self, problem: any):
        """
        Maps the problem dict to a quadratic program.

        :param problem: Problem formulation in dict form
        :type problem: dict
        :return: quadratic program in qiskit-optimization format
        :rtype: QuadraticProgram
        """
        # Details at:
        # https://coin-or.github.io/pulp/guides/how_to_export_models.html
        # https://qiskit.org/documentation/stable/0.26/tutorials/optimization/2_converters_for_quadratic_programs.html
        qp = QuadraticProgram()

        # Variables
        for variable_dict in problem["variables"]:
            if variable_dict["cat"] == "Integer":
                lb = variable_dict["lowBound"]
                ub = variable_dict["upBound"]
                name = variable_dict["name"]
                # If the integer variable is actually a binary variable
                if lb == 0 and ub == 1:
                    qp.binary_var(name)
                # If the integer variable is non-binary
                else:
                    qp.integer_var(lowerbound=lb, upperbound=ub, name=name)

        # Objective function
        # Arguments:
        obj_arguments = {}
        for arg in problem["objective"]["coefficients"]:
            obj_arguments[arg["name"]] = arg["value"]

        # Maximize
        if problem["parameters"]["sense"] == -1:
            qp.maximize(linear=obj_arguments)
        # Minimize
        else:
            qp.minimize(linear=obj_arguments)

        # Constraints
        for constraint in problem["constraints"]:
            const_arguments = {}
            for arg in constraint["coefficients"]:
                const_arguments[arg["name"]] = arg["value"]
            sense = constraint["sense"]
            if sense == -1:
                const_sense = "LE"
            elif sense == 1:
                const_sense = "GE"
            else:
                const_sense = "E"
            qp.linear_constraint(linear=const_arguments, sense=const_sense, rhs=-1 * constraint["constant"],
                                 name=constraint["name"])
        return qp

    def map(self, problem: any, config: Config) -> (dict, float):
        """
        Use Ising mapping of qiskit-optimize
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with the Ising, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()

        # Map Linear problem from dictionary (generated by pulp) to quadratic program
        qp = self.map_pulp_to_qiskit(problem)
        # print(qp.prettyprint())
        logging.info(qp.export_as_lp_string())

        # convert quadratic problem to qubo to ising
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(qp)
        # get variables
        variables = []
        for variable in qubo.variables:
            variables.append(variable.name)

        qubitOp, _ = qubo.to_ising()

        self.global_variables = variables

        # reverse generate J and t out of qubit PauliSumOperator from qiskit
        t_matrix = np.zeros(qubitOp.num_qubits, dtype=complex)
        j_matrix = np.zeros((qubitOp.num_qubits, qubitOp.num_qubits), dtype=complex)

        for i in qubitOp:
            pauli_str, coeff = i.primitive.to_list()[0]
            logging.info((pauli_str, coeff))
            pauli_str_list = list(pauli_str)
            index_pos_list = list(locate(pauli_str_list, lambda a: a == 'Z'))
            if len(index_pos_list) == 1:
                # update t
                t_matrix[index_pos_list[0]] = coeff
            elif len(index_pos_list) == 2:
                j_matrix[index_pos_list[0]][index_pos_list[1]] = coeff

        return {"J": j_matrix, "t": t_matrix}, end_time_measurement(start)

    def reverse_map(self, solution: dict) -> (dict, float):
        """
        Maps the solution back to the representation needed by the ACL class for validation/evaluation.

        :param solution: bit_string containing the solution
        :type solution: dict
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()
        if np.any(solution == "-1"):  # ising model output from Braket QAOA
            solution = self._convert_ising_to_qubo(solution)
        result = {"status": [0]}
        variables = {}
        objective_value = 0
        for bit in solution:
            if solution[bit] > 0:
                # We only care about assignments:
                if "x" in self.global_variables[bit]:
                    variables[self.global_variables[bit]] = solution[bit]
                    result["status"] = 'Optimal'
                    objective_value += solution[bit]
        result["variables"] = variables
        result["obj_value"] = objective_value
        return result, end_time_measurement(start)

    @staticmethod
    def _convert_ising_to_qubo(solution: any) -> any:
        solution = np.array(solution)
        with np.nditer(solution, op_flags=['readwrite']) as it:
            for x in it:
                if x == -1:
                    x[...] = 0
        return solution

    def get_default_submodule(self, option: str) -> Core:
        if option == "QAOA":
            from modules.solvers.QAOA import QAOA  # pylint: disable=C0415
            return QAOA()
        elif option == "QiskitQAOA":
            from modules.solvers.QiskitQAOA import QiskitQAOA  # pylint: disable=C0415
            return QiskitQAOA()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")