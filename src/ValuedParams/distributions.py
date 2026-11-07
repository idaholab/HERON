# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Declares the probability distributions HERON supports in the <uncertainty> ValuedParam.
  HERON chooses which distributions it supports rather than loading every RAVEN Distribution.
  Each declared distribution uses the same node name, attributes, and parameter nodes as the
  RAVEN Distribution of the same name, so the user's XML can be passed to RAVEN unchanged.
"""
from dataclasses import dataclass

from ravenframework.utils import InputData, InputTypes

@dataclass(frozen=True)
class Param:
  """
    A parameter node of a distribution, e.g. <mean> for the Normal distribution.
  """
  name: str            # XML node name
  descr: str           # LaTeX description for the user manual
  required: bool       # True if the distribution needs this parameter (enforced by RAVEN, noted in docs)

@dataclass(frozen=True)
class DistributionSpec:
  """
    A distribution HERON supports under <uncertainty>.
  """
  name: str          # XML node name, matching the RAVEN Distribution
  descr: str         # LaTeX description for the user manual
  params: tuple = () # parameter nodes (Param), in addition to the bounds

# attributes and bound nodes shared by every distribution
_name_descr = r"""unique name for this distribution. HERON replaces it when writing the RAVEN workflow."""
_verbosity_options = ('silent', 'quiet', 'all', 'debug')
_verbosity_descr = r"""verbosity of messages from this distribution while RAVEN runs."""
_lower_descr = r"""minimum allowed value. For distributions that are not otherwise bounded below, this truncates
    the distribution and renormalizes its probability."""
_upper_descr = r"""maximum allowed value. For distributions that are not otherwise bounded above, this truncates
    the distribution and renormalizes its probability."""

SUPPORTED = (
  DistributionSpec('Uniform',
    r"""uniform distribution with probability density $f(x)=\frac{1}{b-a}$ for $a \le x \le b$, where
        $a$ is \xmlNode{lowerBound} and $b$ is \xmlNode{upperBound}. Both bounds are required."""),
  DistributionSpec('Normal',
    r"""normal (Gaussian) distribution with probability density
        $f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$.""",
    (Param('mean', r"""mean $\mu$ of the distribution.""", True),
     Param('sigma', r"""standard deviation $\sigma$ of the distribution.""", True))),
  DistributionSpec('LogNormal',
    r"""log-normal distribution with probability density
        $f(x)=\frac{1}{(x-x_{0})\sigma\sqrt{2\pi}}e^{-\frac{(\ln{(x-x_{0})}-\mu)^2}{2\sigma^2}}$ for $x > x_{0}$.""",
    (Param('mean', r"""mean $\mu$ of the logarithm of the variable.""", True),
     Param('sigma', r"""standard deviation $\sigma$ of the logarithm of the variable.""", True),
     Param('low', r"""lower boundary $x_{0}$ of the distribution's domain. \default{0.0}""", False))),
  DistributionSpec('Triangular',
    r"""triangular distribution with probability density $f(x)=\frac{2(x-a)}{(b-a)(c-a)}$ for $a \le x < c$,
        $f(x)=\frac{2(b-x)}{(b-a)(b-c)}$ for $c \le x \le b$, and zero otherwise. Truncation is not supported:
        if \xmlNode{lowerBound} or \xmlNode{upperBound} is given, it must equal \xmlNode{min} or \xmlNode{max}.""",
    (Param('min', r"""lower boundary $a$ of the distribution.""", True),
     Param('max', r"""upper boundary $b$ of the distribution.""", True),
     Param('apex', r"""location $c$ of the peak of the distribution.""", True))),
  DistributionSpec('Beta',
    r"""beta distribution with probability density
        $f(x)=\frac{y^{\alpha-1}(1-y)^{\beta-1}}{(x_{1}-x_{0})B(\alpha,\beta)}$ for $0 \le y \le 1$, where
        $y=\frac{x-x_{0}}{x_{1}-x_{0}}$ and $B$ is the beta function. Give either both \xmlNode{alpha} and
        \xmlNode{beta}, or only \xmlNode{peakFactor}.""",
    (Param('low', r"""lower boundary $x_{0}$ of the distribution. \default{0.0}""", False),
     Param('high', r"""upper boundary $x_{1}$ of the distribution. \default{1.0}""", False),
     Param('alpha', r"""first shape parameter $\alpha$.""", False),
     Param('beta', r"""second shape parameter $\beta$.""", False),
     Param('peakFactor', r"""alternative to \xmlNode{alpha} and \xmlNode{beta} for a symmetric distribution,
         from 0 to 1. Larger values give a sharper peak.""", False))),
  DistributionSpec('Gamma',
    r"""gamma distribution with probability density
        $f(x)=\frac{\beta^{\alpha}(x-x_{0})^{\alpha-1}e^{-\beta (x-x_{0})}}{\Gamma(\alpha)}$ for $x \ge x_{0}$,
        where $\Gamma$ is the gamma function.""",
    (Param('low', r"""lower boundary $x_{0}$ of the distribution's domain. \default{0.0}""", False),
     Param('alpha', r"""shape parameter $\alpha$.""", True),
     Param('beta', r"""rate parameter $\beta$ (the inverse of the scale). \default{1.0}""", False))),
  DistributionSpec('Exponential',
    r"""exponential distribution with probability density $f(x)=\lambda e^{-\lambda (x-x_{0})}$ for $x \ge x_{0}$.""",
    (Param('lambda', r"""rate parameter $\lambda$.""", True),
     Param('low', r"""lower boundary $x_{0}$ of the distribution's domain. \default{0.0}""", False))),
  DistributionSpec('Weibull',
    r"""Weibull distribution with probability density
        $f(x)=\frac{k}{\lambda}\left(\frac{x-x_{0}}{\lambda}\right)^{k-1}e^{-((x-x_{0})/\lambda)^k}$ for $x \ge x_{0}$.""",
    (Param('k', r"""shape parameter $k$.""", True),
     Param('lambda', r"""scale parameter $\lambda$.""", True),
     Param('low', r"""lower boundary $x_{0}$ of the distribution's domain. \default{0.0}""", False))),
  DistributionSpec('LogUniform',
    r"""log-uniform (reciprocal) distribution: the logarithm of the variable is uniformly distributed
        between \xmlNode{lowerBound} and \xmlNode{upperBound}, which are therefore given as exponents of
        \xmlNode{base}. For example, with \xmlNode{base} \texttt{decimal}, bounds of 1 and 3 give values
        from 10 to 1000. Both bounds are required.""",
    (Param('base', r"""base of the logarithm: \texttt{natural} ($e$) or \texttt{decimal} (10).""", True),)),
  DistributionSpec('Logistic',
    r"""logistic distribution with probability density
        $f(x)=\frac{e^{-(x-\mu)/s}}{s\left(1+e^{-(x-\mu)/s}\right)^2}$.""",
    (Param('location', r"""location parameter $\mu$.""", True),
     Param('scale', r"""scale parameter $s$.""", True))),
  DistributionSpec('Laplace',
    r"""Laplace distribution with probability density $f(x)=\frac{1}{2b}e^{-\frac{|x-\mu|}{b}}$.""",
    (Param('location', r"""location parameter $\mu$.""", True),
     Param('scale', r"""scale parameter $b$.""", True))),
)

def supported_names():
  """
    Gets the names of the distributions HERON supports.
    @ In, None
    @ Out, names, list(str), distribution node names
  """
  return [dist.name for dist in SUPPORTED]

def make_spec(dist):
  """
    Builds the input specification for one supported distribution.
    @ In, dist, DistributionSpec, the distribution declaration
    @ Out, spec, InputData.ParameterInput, input specification for the distribution node
  """
  spec = InputData.parameterInputFactory(dist.name, descr=dist.descr)
  spec.addParam('name', param_type=InputTypes.StringType, required=True, descr=_name_descr)
  verbosity = InputTypes.makeEnumType('verbosity', 'verbosityType', list(_verbosity_options))
  spec.addParam('verbosity', param_type=verbosity, descr=_verbosity_descr)
  spec.addSub(InputData.parameterInputFactory('lowerBound', contentType=InputTypes.FloatType, descr=_lower_descr))
  spec.addSub(InputData.parameterInputFactory('upperBound', contentType=InputTypes.FloatType, descr=_upper_descr))
  for param in dist.params:
    # TODO LogUniform <base> should only allow "natural" or "decimal". RAVEN's spec means to enforce that,
    # but passes the enum type in the wrong argument, so any text parses and a bad value only fails once
    # RAVEN reads the workflow. It is left untyped here to match RAVEN.
    if (dist.name, param.name) == ('LogUniform', 'base'):
      content = None
    else:
      content = InputTypes.FloatType
    descr = param.descr if not param.required else f'{param.descr} Required.'
    spec.addSub(InputData.parameterInputFactory(param.name, contentType=content, descr=descr))
  return spec
