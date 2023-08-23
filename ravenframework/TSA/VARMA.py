# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  AutoRegressive Moving Average time series analysis
"""
import os
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .. import Distributions

from ..utils import InputData, InputTypes, randomUtils, xmlUtils, mathUtils, importerUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())

from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesCharacterizer, TimeSeriesTransformer


class VARMA(TimeSeriesGenerator, TimeSeriesCharacterizer, TimeSeriesTransformer):
  r"""
    AutoRegressive Moving Average time series analyzer algorithm
  """
  # class attribute
  ## define the clusterable features for this trainer.
  _features = ['ar',
               'ma',
               'cov',
               'const']
  _acceptsMissingValues = True
  _isStochastic = True

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(VARMA, cls).getInputSpecification()
    specs.name = 'varma'
    specs.description = r"""characterizes the vector-valued signal using Auto-Regressive and Moving
        Average coefficients to stochastically fit the training signal.
        The ARMA representation has the following form:
        \begin{equation*}
          A_t = \sum_{i=1}^P \phi_i A_{t-i} + \varepsilon_t + \sum_{j=1}^Q \theta_j \varepsilon_{t-j},
        \end{equation*}
        where $t$ indicates a discrete time step, $\phi$ are the signal lag (or auto-regressive)
        coefficients, $P$ is the number of signal lag terms to consider, $\varepsilon$ is a random noise
        term, $\theta$ are the noise lag (or moving average) coefficients, and $Q$ is the number of
        noise lag terms to consider. For signal $\A_t$ which is a $k \times 1$ vector, each $\phi_i$
        and $\theta_j$ are $k \times k$ matrices, and $\varepsilon_t$ is characterized by the
        $k \times k$ covariance matrix $\Sigma$. The VARMA algorithms are developed in RAVEN using the
        \texttt{statsmodels} Python library."""
    specs.addParam('reduce_memory', param_type=InputTypes.BoolType, required=False,
                   descr=r"""activates a lower memory usage VARMA training. This does tend to result
                         in a slightly slower training time, at the benefit of lower memory usage.
                         Note that the VARMA must be retrained to change this property; it cannot be
                         applied to serialized models.""", default=False)
    specs.addSub(InputData.parameterInputFactory('P', contentType=InputTypes.IntegerType,
                 descr=r"""the number of terms in the AutoRegressive term to retain in the
                       regression; typically represented as $P$ in literature."""))
    specs.addSub(InputData.parameterInputFactory('Q', contentType=InputTypes.IntegerType,
                 descr=r"""the number of terms in the Moving Average term to retain in the
                       regression; typically represented as $Q$ in literature."""))
    return specs

  #
  # API Methods
  #
  def __init__(self, *args, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, args, list, an arbitrary list of positional values
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    # general infrastructure
    super().__init__(*args, **kwargs)

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['P'] = spec.findFirst('P').value
    settings['Q'] = spec.findFirst('Q').value
    settings['reduce_memory'] = spec.parameterValues.get('reduce_memory', settings['reduce_memory'])
    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    if 'engine' not in settings:
      settings['engine'] = randomUtils.newRNG()
    if 'low_memory' not in settings:
      settings['reduce_memory'] = False
    return settings

  def fit(self, signal, pivot, targets, settings):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, settings for this ROM
      @ Out, params, dict, characteristic parameters
    """
    # lazy statsmodels import
    import statsmodels.api

    P = settings['P']
    Q = settings['Q']
    seed = settings['seed']
    if seed is not None:
      randomUtils.randomSeed(seed, engine=settings['engine'], seedBoth=True)

    model = statsmodels.api.tsa.VARMAX(endog=signal, order=(P, Q))
    results = model.fit(disp=False, maxiter=1000)

    lenHist, numVars = signal.shape
    # train multivariate normal distributions using covariances, keep it around so we can control the RNG
    ## it appears "measurement" always has 0 covariance, and so is all zeros (see _generateVARMASignal)
    ## all the noise comes from the stateful properties
    stateDist = self._trainMultivariateNormal(numVars, np.zeros(numVars), model.ssm['state_cov'])
    # train initial state sampler
    ## Used to pick an initial state for the VARMA by sampling from the multivariate normal noise
    #    and using the AR and MA initial conditions.  Implemented so we can control the RNG internally.
    #    Implementation taken directly from statsmodels.tsa.statespace.kalman_filter.KalmanFilter.simulate
    ## get mean
    smoother = model.ssm
    mean = np.linalg.solve(np.eye(smoother.k_states) - smoother['transition',:,:,0],
                          smoother['state_intercept',:,0])
    ## get covariance
    r = smoother['selection',:,:,0]
    q = smoother['state_cov',:,:,0]
    selCov = r.dot(q).dot(r.T)
    cov = solve_discrete_lyapunov(smoother['transition',:,:,0], selCov)
    # FIXME it appears this is always resulting in a lowest-value initial state.  Why?
    initDist = self._trainMultivariateNormal(len(mean),mean,cov)
    params = {}
    params['VARMA'] = {'model': results,
                       'targets': targets,
                       'initDist': initDist,
                       'noiseDist': stateDist}
    return params

  def _trainMultivariateNormal(self, dim, means, cov):
    """
      Trains multivariate normal distribution for future sampling
      @ In, dim, int, number of dimensions
      @ In, means, np.array, distribution mean
      @ In, cov, np.ndarray, dim x dim matrix of covariance terms
      @ Out, dist, Distributions.MultivariateNormal, distribution
    """
    dist = Distributions.MultivariateNormal()
    dist.method = 'pca'
    dist.dimension = dim
    dist.rank = dim
    dist.mu = means
    dist.covariance = np.ravel(cov)
    dist.initializeDistribution()
    dist.workingDir = os.getcwd()
    return dist

  def getResidual(self, initial, params, pivot, settings):
    """
      Removes trained signal from data and find residual
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, residual, np.array, reduced signal shaped [pivotValues, targets]
    """
    residual = params['VARMA']['model'].resid
    return residual

  def getComposite(self, initial, params, pivot, settings):
    """
      Combines two component signals to form a composite signal. This is essentially the inverse
      operation of the getResidual method.
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, composite, np.array, resulting composite signal
    """
    # Add a generated ARMA signal to the initial signal.
    synthetic = self.generate(params, pivot, settings)
    composite = initial + synthetic
    return composite

  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, settings for this ROM
      @ Out, synthetic, np.array(float), synthetic ARMA signal
    """
    model = params['VARMA']['model']
    numSamples = len(pivot)
    numVars = model.model.k_endog

    # sample measure, state shocks
    measureShocks = np.zeros([numSamples, numVars])

    ## state shocks come from sampling multivariate, with CROW
    noiseDist = params['VARMA']['noiseDist']
    initDist = params['VARMA']['initDist']
    stateShocks = np.array([noiseDist.rvs() for _ in range(numSamples)])

    # pick an intial by sampling multinormal distribution
    init = np.array(initDist.rvs())
    synthetic = model.simulate(numSamples,
                               initial_state=init,
                               measurement_shocks=measureShocks,
                               state_shocks=stateShocks)
    return synthetic

  def getParamNames(self, settings):
    """
      Return list of expected variable names based on the parameters
      @ In, settings, dict, training parameters for this algorithm
      @ Out, names, list, string list of names
    """
    # TODO
    names = []
    base = f'{self.name}'
    for target in settings['target']:
      base = f'{self.name}__{target}'
      names.append(f'{base}__constant')
      names.append(f'{base}__covariance')
      for p in range(settings['P']):
        names.append(f'{base}__AR__{p}')
      for q in range(settings['Q']):
        names.append(f'{base}__MA__{q}')
    return names

  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from characterize)
      @ Out, rlz, dict, realization-style response
    """
    # TODO
    rlz = {}
    for target, info in params.items():
      base = f'{self.name}__{target}'
      rlz[f'{base}__constant'] = info['arma']['const']
      rlz[f'{base}__variance'] = info['arma']['var']
      for p, ar in enumerate(info['arma']['ar']):
        rlz[f'{base}__AR__{p}'] = ar
      for q, ma in enumerate(info['arma']['ma']):
        rlz[f'{base}__MA__{q}'] = ma
    return rlz

  # clustering
  def getClusteringValues(self, nameTemplate: str, requests: list, params: dict) -> dict:
    """
      Provide the characteristic parameters of this ROM for clustering with other ROMs
      @ In, nameTemplate, str, formatting string template for clusterable params (target, metric id)
      @ In, requests, list, list of requested attributes from this ROM
      @ In, params, dict, parameters from training this ROM
      @ Out, features, dict, params as {paramName: value}
    """
    # nameTemplate convention:
    # -> target is the trained variable (e.g. Signal, Temperature)
    # -> metric is the algorithm used (e.g. Fourier, ARMA)
    # -> id is the subspecific characteristic ID (e.g. sin, AR_0)
    features = {}
    for target, info in params.items():
      data = info['arma']
      if 'ar' in requests:
        for p, phi in enumerate(data['ar']):
          key = nameTemplate.format(target=target, metric=self.name, id=f'ar_{p}')
          features[key] = phi
      if 'ma' in requests:
        for q, theta in enumerate(data['ma']):
          key = nameTemplate.format(target=target, metric=self.name, id=f'ma_{q}')
          features[key] = theta
      if 'const' in requests:
        key = nameTemplate.format(target=target, metric=self.name, id='const')
        features[key] = data['const']
      if 'var' in requests:
        key = nameTemplate.format(target=target, metric=self.name, id='var')
        features[key] = data['var']
    return features

  def setClusteringValues(self, fromCluster, params):
    """
      Interpret returned clustering settings as settings for this algorithm.
      Acts somewhat as the inverse of getClusteringValues.
      @ In, fromCluster, list(tuple), (target, identifier, values) to interpret as settings
      @ In, params, dict, trained parameter settings
      @ Out, params, dict, updated parameter settings
    """
    for target, identifier, value in fromCluster:
      value = float(value)
      if identifier in ['const', 'var']:
        params[target]['arma'][identifier] = value
      elif identifier.startswith('ar_'):
        index = int(identifier.split('_')[1])
        params[target]['arma']['ar'][index] = value
      elif identifier.startswith('ma_'):
        index = int(identifier.split('_')[1])
        params[target]['arma']['ma'][index] = value
    return params

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.characterize
      @ Out, None
    """
    model = params['VARMA']['model']
    targets = params['VARMA']['targets']
    nodes = self._parseParams(model.param_names, model.params, targets)
    for name, node in nodes.items():
      writeTo.append(node)

  def _parseParams(self, names, values, targetNames):
    """
      Parses VARMA parameter names into XML nodes
      @ In, names, list(str), parameter names
      @ In, values, list(float), parameter values
      @ in, targetNames, list(str), list of target variable names
      @ Out, nodes, dict, dictionary of XML nodes
    """
    nodes = {
      'constant': xmlUtils.newNode('constant'),
      'covariance': xmlUtils.newNode('covariance'),
      'AR': xmlUtils.newNode('AR'),
      'MA': xmlUtils.newNode('MA')
    }

    for name, value in zip(names, values):
      parts = name.split('.')  # e.g. 'intercept.y1', 'sqrt.var.y1', 'L1.y1.y1', 'L1.e(y1).y2', 'sqrt.cov.y1.y2'
      if parts[0] == 'intercept':
        target = targetNames[int(parts[1][1:]) - 1]
        nodes['constant'].append(xmlUtils.newNode(target, text=f'{value}'))
      elif parts[1] == 'var':  # variance (diagonal)
        target = targetNames[int(parts[2][1:]) - 1]
        nodes['covariance'].append(xmlUtils.newNode(f'var_{target}', text=f'{value ** 2}'))
      elif parts[1] == 'cov':  # covariance (off-diagonal)
        target1 = targetNames[int(parts[2][1:]) - 1]
        target2 = targetNames[int(parts[3][1:]) - 1]
        nodes['covariance'].append(xmlUtils.newNode(f'cov_{target1}_{target2}', text=f'{value ** 2}'))
      elif parts[0].startswith('L') and parts[1].startswith('y'):  # AR coeff
        target1 = targetNames[int(parts[1][1:]) - 1]
        target2 = targetNames[int(parts[2][1:]) - 1]
        if target1 == target2:
          nodes['AR'].append(xmlUtils.newNode(f'Lag{parts[0][1:]}_{target1}', text=f'{value}'))
        else:
          nodes['AR'].append(xmlUtils.newNode(f'Lag{parts[0][1:]}_{target1}_{target2}', text=f'{value}'))
      elif parts[0].startswith('L') and parts[1].startswith('e'):  # MA coeff
        target1 = targetNames[int(parts[1][3:-1]) - 1]
        target2 = targetNames[int(parts[2][1:]) - 1]
        if target1 == target2:
          nodes['MA'].append(xmlUtils.newNode(f'Lag{parts[0][1:]}_{target1}', text=f'{value}'))
        else:
          nodes['MA'].append(xmlUtils.newNode(f'Lag{parts[0][1:]}_{target1}_{target2}', text=f'{value}'))
      else:
        raise ValueError(f'Unrecognized parameter name "{name}"!')

    return nodes
