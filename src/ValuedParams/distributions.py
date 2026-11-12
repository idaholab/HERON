# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Declares the probability distributions HERON supports in the <uncertainty> ValuedParam.
  HERON chooses which distributions it supports rather than loading every RAVEN Distribution.
  Each declared distribution uses the same node name, attributes, and parameter nodes as the
  RAVEN Distribution of the same name, so the user's XML can be passed to RAVEN unchanged.
  Only one-dimensional distributions are supported, since each <uncertainty> is a single value.
"""
from dataclasses import dataclass

from ravenframework.utils import InputData, InputTypes

@dataclass(frozen=True)
class Attr:
  """
    An attribute of a distribution node or one of its parameter nodes.
  """
  name: str        # XML attribute name
  type: type       # InputTypes type of the value
  required: bool   # True if the attribute must be given (enforced when parsing)
  descr: str       # LaTeX description for the user manual

@dataclass(frozen=True)
class Param:
  """
    A parameter node of a distribution, e.g. <mean> for the Normal distribution.
  """
  name: str                             # XML node name
  descr: str                            # LaTeX description for the user manual
  required: bool                        # True if the distribution needs this parameter (enforced by RAVEN, noted in docs)
  content: type = InputTypes.FloatType  # InputTypes type of the node's text, or None for no type
  attrs: tuple = ()                     # attributes (Attr) of this node
  quantity: InputData.Quantity = InputData.Quantity.zero_to_infinity # occurrences allowed; only checked when ordered
  default: object = 'no-default'        # default shown in the user manual, as in InputData

@dataclass(frozen=True)
class DistributionSpec:
  """
    A distribution HERON supports under <uncertainty>.
  """
  name: str              # XML node name, matching the RAVEN Distribution
  descr: str             # LaTeX description for the user manual
  params: tuple = ()     # parameter nodes (Param), in addition to the bounds
  ordered: bool = False  # True if the parameter nodes must appear in the declared order
  bounds: bool = True    # True if the distribution accepts <lowerBound> and <upperBound>
  verbosity: bool = True # True if the distribution accepts the "verbosity" attribute

# attributes and bound nodes shared by distributions
_name_attr = Attr('name', InputTypes.StringType, True,
    r"""unique name for this distribution. HERON replaces it when writing the RAVEN workflow.""")
_verbosity_attr = Attr('verbosity', InputTypes.makeEnumType('verbosity', 'verbosityType', ['silent', 'quiet', 'all', 'debug']),
    False, r"""verbosity of messages from this distribution while RAVEN runs.""")
_bound_params = (
  Param('lowerBound', r"""minimum allowed value. For distributions that are not otherwise bounded below, this truncates
    the distribution and renormalizes its probability.""", False),
  Param('upperBound', r"""maximum allowed value. For distributions that are not otherwise bounded above, this truncates
    the distribution and renormalizes its probability.""", False),
)

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
    # TODO <base> should only allow "natural" or "decimal". RAVEN's spec means to enforce that, but passes
    # the enum type in the wrong argument, so any text parses and a bad value only fails once RAVEN reads
    # the workflow. It is left untyped here to match RAVEN.
    (Param('base', r"""base of the logarithm: \texttt{natural} ($e$) or \texttt{decimal} (10).""", True,
           content=None),)),
  DistributionSpec('Logistic',
    r"""logistic distribution with probability density
        $f(x)=\frac{e^{-(x-\mu)/s}}{s\left(1+e^{-(x-\mu)/s}\right)^2}$.""",
    (Param('location', r"""location parameter $\mu$.""", True),
     Param('scale', r"""scale parameter $s$.""", True))),
  DistributionSpec('Laplace',
    r"""Laplace distribution with probability density $f(x)=\frac{1}{2b}e^{-\frac{|x-\mu|}{b}}$.""",
    (Param('location', r"""location parameter $\mu$.""", True),
     Param('scale', r"""scale parameter $b$.""", True))),
  DistributionSpec('Custom1D',
    r"""distribution defined by a tabulated probability density or cumulative distribution function in a
        CSV file.""",
    (Param('workingDir', r"""directory containing \xmlNode{dataFilename}, relative to the working directory.""",
           False, content=InputTypes.StringType),
     Param('functionType', r"""kind of function tabulated in the file: \texttt{pdf} or \texttt{cdf}.""", True,
           content=InputTypes.StringType),
     Param('dataFilename', r"""name of the CSV file.""", True, content=InputTypes.StringType),
     Param('functionID', r"""column of the file holding the function values.""", True,
           content=InputTypes.StringType),
     Param('variableID', r"""column of the file holding the variable values.""", True,
           content=InputTypes.StringType))),
  # discrete distributions
  DistributionSpec('Poisson',
    r"""Poisson distribution with probability mass function $f(k)=\frac{\mu^k e^{-\mu}}{k!}$ for
        $k=0,1,2,\ldots$. Truncation is not supported.""",
    (Param('mu', r"""mean $\mu$ of the distribution.""", True),)),
  DistributionSpec('Binomial',
    r"""binomial distribution with probability mass function $f(k)=\binom{n}{k}p^k(1-p)^{n-k}$ for
        $k=0,1,\ldots,n$. Truncation is not supported.""",
    (Param('n', r"""number of trials $n$.""", True, content=InputTypes.IntegerType),
     Param('p', r"""probability of success $p$ in each trial.""", True))),
  DistributionSpec('Bernoulli',
    r"""Bernoulli distribution, which takes the value 1 with probability $p$ and 0 with probability $1-p$.
        Truncation is not supported.""",
    (Param('p', r"""probability $p$ of the value 1.""", True),)),
  DistributionSpec('Geometric',
    r"""geometric distribution of the number of independent trials, each with success probability $p$,
        needed to obtain a success. Truncation is not supported.""",
    (Param('p', r"""probability of success $p$ in each trial.""", True),)),
  DistributionSpec('UniformDiscrete',
    r"""discrete distribution in which each of $N$ values is equally likely. The values are either listed
        in \xmlNode{values}, or spaced evenly from \xmlNode{lowerBound} to \xmlNode{upperBound} (both
        required in that case) using either \xmlNode{nPoints} or \xmlNode{step}.""",
    (Param('nPoints', r"""number of values from \xmlNode{lowerBound} to \xmlNode{upperBound}.""", False,
           content=InputTypes.IntegerType),
     Param('step', r"""spacing between values from \xmlNode{lowerBound} to \xmlNode{upperBound}. It must
         evenly divide the range.""", False),
     Param('values', r"""comma-separated list of the values. Cannot be combined with the bounds,
         \xmlNode{nPoints}, or \xmlNode{step}.""", False, content=InputTypes.FloatListType),
     # TODO <strategy> should only allow "withReplacement" or "withoutReplacement". RAVEN's spec means to
     # enforce that, but passes the enum type in the wrong argument, so any text parses and a bad value
     # only fails once RAVEN reads the workflow. It is left untyped here to match RAVEN.
     Param('strategy', r"""sampling strategy: \texttt{withReplacement} or \texttt{withoutReplacement}. With
         replacement, every sample is drawn from all $N$ values; without replacement, values already sampled
         are excluded until all values have been drawn.""", True, content=None))),
  DistributionSpec('Categorical',
    r"""discrete distribution over a finite set of outcomes, each with its own probability. Outcomes may be
        numbers or strings. The \xmlNode{state} nodes must come before \xmlNode{rtol}.""",
    (Param('state', r"""probability of one outcome. Give one \xmlNode{state} node per outcome.""", True,
           attrs=(Attr('outcome', InputTypes.FloatOrStringType, True, r"""value of this outcome."""),),
           quantity=InputData.Quantity.one_to_infinity),
     Param('rtol', r"""relative tolerance for identifying numeric outcomes as equal. Not used for string
         outcomes.""", False, default=1e-6)),
    ordered=True, bounds=False, verbosity=False),
  DistributionSpec('MarkovCategorical',
    r"""discrete distribution over a finite set of outcomes whose probabilities are the steady-state
        probabilities of a Markov model with the given transition matrix. The \xmlNode{state} nodes must come
        before \xmlNode{transition}, which must come before \xmlNode{workingDir}.""",
    (Param('state', r"""one outcome of the distribution. Give one \xmlNode{state} node per outcome.""", True,
           content=InputTypes.StringType,
           attrs=(Attr('outcome', InputTypes.FloatType, True, r"""value of this outcome."""),
                  Attr('index', InputTypes.IntegerType, True,
                       r"""index of this outcome's state in the transition matrix, starting from 1.""")),
           quantity=InputData.Quantity.one_to_infinity),
     Param('transition', r"""transition matrix of the Markov model, as space-separated values in row order.
         It must be square, with at least two rows.""", True, content=InputTypes.StringType,
           quantity=InputData.Quantity.zero_to_one),
     Param('workingDir', r"""working directory for the distribution.""", False, content=InputTypes.StringType)),
    ordered=True, bounds=False, verbosity=False),
)

def make_spec(dist):
  """
    Builds the input specification for one supported distribution.
    @ In, dist, DistributionSpec, the distribution declaration
    @ Out, spec, InputData.ParameterInput, input specification for the distribution node
  """
  spec = InputData.parameterInputFactory(dist.name, ordered=dist.ordered, descr=dist.descr)
  _add_attrs(spec, (_name_attr, _verbosity_attr) if dist.verbosity else (_name_attr,))
  params = (_bound_params if dist.bounds else ()) + dist.params
  for param in params:
    descr = param.descr if not param.required else f'{param.descr} Required.'
    sub = InputData.parameterInputFactory(param.name, contentType=param.content, descr=descr, default=param.default)
    _add_attrs(sub, param.attrs)
    spec.addSub(sub, quantity=param.quantity)
  return spec

def _add_attrs(spec, attrs):
  """
    Adds attribute declarations to a spec.
    @ In, spec, InputData.ParameterInput, spec to add the attributes to
    @ In, attrs, tuple(Attr), attributes to add
    @ Out, None
  """
  for attr in attrs:
    spec.addParam(attr.name, param_type=attr.type, required=attr.required, descr=attr.descr)
