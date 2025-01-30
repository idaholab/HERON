# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Optimization features

@author: Jacob Bryan (@j-bryan)
@date: 2024-11-08
"""
from ..xml_utils import find_node
from .base import RavenSnippet
from .samplers import Sampler
from .dataobjects import DataObject


# Inheriting from Sampler mimics RAVEN inheritance structure
class Optimizer(Sampler):
  """ A base class for RAVEN optimizer XML snippets """
  snippet_class = "Optimizers"

  def set_opt_settings(self, opt_settings: dict) -> None:
    """
    Set optimizer settings
    @ In, opt_settings, dict, the optimizer settings
    @ Out, None
    """
    # convergence settings
    convergence = find_node(self, "convergence")
    for k, v in opt_settings.get("convergence", {}).items():
      find_node(convergence, k).text = v

    # persistence
    if "persistence" in opt_settings:
      find_node(convergence, "persistence").text = opt_settings["persistence"]

    # samplerInit settings
    sampler_init = find_node(self, "samplerInit")
    for name in opt_settings.keys() & {"limit", "type"}:  # writeEvery not exposed in HERON input
      find_node(sampler_init, name).text = opt_settings[name]

  @property
  def objective(self) -> str | None:
    """
    Optimization objective getter
    @ In, None
    @ Out, objective, str | None, the objective
    """
    node = self.find("objective")
    return None if node is None else node.text

  @objective.setter
  def objective(self, value: str) -> None:
    """
    Optimization objective getter
    @ In, value, str, the objective
    @ Out, None
    """
    find_node(self, "objective").text = value

  @property
  def target_evaluation(self) -> str | None:
    """
    Target evaluation getter
    @ In, None
    @ Out, target_evaluation, str | None, the target evaluation data object name
    """
    node = self.find("TargetEvaluation")
    return None if node is None else node.text

  @target_evaluation.setter
  def target_evaluation(self, value: DataObject) -> None:
    """
    Target evaluation setter
    @ In, value, DataObject, the target evaluation data object
    @ Out, None
    """
    if not getattr(value, "snippet_class", None) == "DataObjects":
      raise TypeError(f"Optimizer evaluation target must be set with a DataObject object. Received '{type(value)}'")
    assemb = value.to_assembler_node("TargetEvaluation")
    # Copy assembler node info over to the existing TargetEvaluation node
    target_eval = self.find("TargetEvaluation")
    if target_eval is None:
      self.append(assemb)
    else:
      target_eval.attrib.update(assemb.attrib)
      target_eval.text = assemb.text

#########################
# Bayesian Optimization #
#########################

class BayesianOptimizer(Optimizer):
  """ Bayesian optimizer snippet class """
  tag = "BayesianOptimizer"

  default_settings = {
    "samplerInit": {
      "limit": 100,
      "type": "max",
      "writeSteps": "every",
      # "initialSeed": ""  # initialSeed not included here so the RAVEN default internal seed is used if not provided
    },
    "ModelSelection": {
      "Duration": 1,
      "Method": "Internal"
    },
    "convergence": {
      "acquisition": 1e-5,
      "persistence": 4
    },
    # NOTE: Acquisition function defaults are handled by the the AcquisitionFunction class. Adding the "Acquisition"
    #       key here just ensures the creation of the <Acquisition> child node.
    "Acquisition": ""
  }

  def __init__(self, name: str | None = None) -> None:
    """
    Constructor
    @ In, name, str, optional, entity name
    @ Out, None
    """
    super().__init__(name, self.default_settings)

  def set_opt_settings(self, opt_settings: dict) -> None:
    """
    Sets optimization settings
    @ In, opt_settings, dict, optimization settings
    @ Out, None
    """
    super().set_opt_settings(opt_settings)

    try:
      bo_settings = opt_settings["algorithm"]["BayesianOpt"]
    except KeyError:  # use defaults
      bo_settings = {}

    # acquisition function
    acquisition = self.find("Acquisition")
    if len(acquisition) > 0:
      acquisition.clear()  # drops all children, tag, text, attribs, etc.
      acquisition.tag = "Acquisition"  # replace cleared tag
    default_acq_func = "ExpectedImprovement"
    acq_func_name = bo_settings.get("acquisition", default_acq_func)
    acq_funcs = {
      "ExpectedImprovement": ExpectedImprovement,
      "ProbabilityOfImprovement": ProbabilityOfImprovement,
      "LowerConfidenceBound": LowerConfidenceBound
    }
    try:
      # FIXME: No acquisition function parameters are exposed to the HERON user
      acq_func = acq_funcs.get(acq_func_name, default_acq_func)()
      acquisition.append(acq_func)
    except KeyError as ke:
      raise ValueError(f"Unrecognized acquisition function {acq_func_name}. Allowed: {acq_funcs.keys()}") from ke

    # random seed
    if "seed" in bo_settings:
      sampler_init = find_node(self, "samplerInit")
      find_node(sampler_init, "initialSeed").text = bo_settings["seed"]

    # modelSelection
    model_selection = find_node(self, "ModelSelection")
    for k, v in bo_settings.get("ModelSelection", {}).items():
      find_node(model_selection, k).text = v

  def set_sampler(self, sampler: Sampler) -> None:
    """
    Set the sampler used for initializing the optimizer
    @ In, sampler, Sampler, the initialization sampler
    @ Out, None
    """
    sampler_node = find_node(self, "Sampler")
    assemb_node = sampler.to_assembler_node("Sampler")
    sampler_node.attrib = assemb_node.attrib
    sampler_node.text = assemb_node.text

  def set_rom(self, rom: RavenSnippet) -> None:
    """
    Set the ROM to be used as a surrogate in the Bayesian optimization
    @ In, rom, RavenSnippet
    """
    if rom.snippet_class != "Models":
      raise TypeError(f"An object which is not a model (type: '{type(rom)}') was provided to the BayesianOptimizer "
                      "to be the ROM.")
    model_node = find_node(self, "ROM")
    assemb_node = rom.to_assembler_node("ROM")
    model_node.attrib = assemb_node.attrib
    model_node.text = assemb_node.text


class ExpectedImprovement(RavenSnippet):
  """ Expected improvement acquisition function """
  tag = "ExpectedImprovement"

  default_settings = {
    "optimizationMethod": "differentialEvolution",
    "seedingCount": 30
  }

  def __init__(self) -> None:
    """
    Constructor
    @ In, None
    @ Out, None
    """
    super().__init__(subelements=self.default_settings)

class ProbabilityOfImprovement(RavenSnippet):
  """ Probability of improvement acquisition function """
  tag = "ProbabilityOfImprovement"

  default_settings = {
    "optimizationMethod": "differentialEvolution",
    "seedingCount": 30,
    "epsilon": 1,
    "rho": 20,
    "transient": "Constant"
  }

  def __init__(self) -> None:
    """
    Constructor
    @ In, None
    @ Out, None
    """
    super().__init__(subelements=self.default_settings)

class LowerConfidenceBound(RavenSnippet):
  """ Lower confidence bound acquisition function """
  tag = "LowerConfidenceBound"

  default_settings = {
    "optimizationMethod": "differentialEvolution",
    "seedingCount": 30,
    "pi": 0.98,
    "transient": "Constant"
  }

  def __init__(self) -> None:
    """
    Constructor
    @ In, None
    @ Out, None
    """
    super().__init__(subelements=self.default_settings)

####################
# Gradient Descent #
####################

class GradientDescent(Optimizer):
  """ Gradient descent optimizer snippet class """
  tag = "GradientDescent"

  default_settings = {
    "samplerInit": {
      "limit": 800,
      "type": "max",
      "writeSteps": "every"
    },
    "gradient": {  # CentralDifference, SPSA not exposed in HERON input
      "FiniteDifference": ""  # gradDistanceScalar option not exposed
    },
    "convergence": {
      "persistence": 1,
      "gradient": 1e-4,
      "objective": 1e-8
    },
    "stepSize": {
      "GradientHistory": {
        "growthFactor": 2,
        "shrinkFactor": 1.5,
        "initialStepScale": 0.2
      }
    },
    "acceptance": {
      "Strict": ""
    }
  }

  def __init__(self, name: str | None = None) -> None:
    """
    Constructor
    @ In, name, str, optional, entity name
    @ Out, None
    """
    super().__init__(name, self.default_settings)

  def set_opt_settings(self, opt_settings: dict) -> None:
    """
    Sets the optimization settings
    @ In, opt_settings, dict, optimization settings
    @ Out, None
    """
    super().set_opt_settings(opt_settings)

    try:
      gd_settings = opt_settings["algorithm"]["GradientDescent"]
    except KeyError:
      return  # nothing else to do

    # stepSize settings
    for name in gd_settings.keys() & {"growthFactor", "shrinkFactor", "initialStepScale"}:
      find_node(self, f"stepSize/GradientHistory/{name}").text = gd_settings[name]
