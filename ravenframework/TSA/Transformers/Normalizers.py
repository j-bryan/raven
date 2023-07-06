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
Created on June 20, 2023
@author: j-bryan

Wrappers for scikit-learn preprocessing scalers.
"""

import sklearn.preprocessing as skl
import numpy as np
from scipy.stats import iqr
from copy import deepcopy

from ..TimeSeriesAnalyzer import TimeSeriesTransformer
from .ScikitLearnBase import SKLTransformer, SKLCharacterizer
from ...utils import InputTypes, xmlUtils


class MaxAbsScaler(SKLCharacterizer):
  """ Wrapper of sklearn.preprocessing.MaxAbsScaler """
  _features = ['scale']
  templateTransformer = skl.MaxAbsScaler()

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'maxabsscaler'
    specs.description = r"""scales the data to the interval $[-1, 1]$. This is done by dividing by
    the largest absolute value of the data."""
    return specs


class MinMaxScaler(SKLCharacterizer):
  """ Wrapper of sklearn.preprocessing.MinMaxScaler """
  _features = ['dataMin', 'dataMax']
  templateTransformer = skl.MinMaxScaler()

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'minmaxscaler'
    specs.description = r"""scales the data to the interval $[0, 1]$. This is done by subtracting the
                        minimum value from each point and dividing by the range."""
    return specs


class RobustScaler(SKLCharacterizer):
  """ Wrapper of sklearn.preprocessing.RobustScaler """
  _features = ['center', 'scale']
  templateTransformer = skl.RobustScaler()

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'robustscaler'
    specs.description = r"""centers and scales the data by subtracting the median and dividing by
    the interquartile range."""
    return specs


class StandardScaler(SKLCharacterizer):
  """ Wrapper of sklearn.preprocessing.StandardScaler """
  _features = ['mean', 'scale']
  templateTransformer = skl.StandardScaler()

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'standardscaler'
    specs.description = r"""centers and scales the data by subtracting the mean and dividing by
    the standard deviation."""
    return specs


class QuantileTransformer(SKLTransformer):
  """ Wrapper of scikit-learn's QuantileTransformer """
  templateTransformer = skl.QuantileTransformer()

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'quantiletransformer'
    specs.description = r"""transforms the data to fit a given distribution by mapping the data to
    a uniform distribution and then to the desired distribution."""
    specs.addParam('nQuantiles', param_type=InputTypes.IntegerType,
                   descr=r"""number of quantiles to use in the transformation. If \xmlAttr{nQuantiles}
                   is greater than the number of data, then the number of data is used instead.""",
                   required=False, default=1000)
    distType = InputTypes.makeEnumType('outputDist', 'outputDistType', ['normal', 'uniform'])
    specs.addParam('outputDistribution', param_type=distType,
                   descr=r"""distribution to transform to. Must be either 'normal' or 'uniform'.""",
                   required=False, default='normal')
    return specs

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['nQuantiles'] = spec.parameterValues.get('nQuantiles', settings['nQuantiles'])
    settings['outputDistribution'] = spec.parameterValues.get('outputDistribution', settings['outputDistribution'])
    self.templateTransformer.set_params(n_quantiles=settings['nQuantiles'],
                                        output_distribution=settings['outputDistribution'])
    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    if 'nQuantiles' not in settings:
      settings['nQuantiles'] = 1000
    if 'outputDistribution' not in settings:
      settings['outputDistribution'] = 'normal'
    return settings


class PeriodicScaler(TimeSeriesTransformer):
  """ Transformer that applies a robust scaling to the data over a certain period """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'periodicscaler'
    specs.description = r"""removes the median and scales the data by the interquartile range for
                            each time step over a given period."""
    specs.addParam('period', param_type=InputTypes.IntegerType, required=True,
                   descr=r"period of the data, i.e. the number of samples in a cycle.")
    specs.addParam('scaling', param_type=InputTypes.makeEnumType('type', 'typeType', ['standard', 'robust']),
                    required=False, default='standard',
                    descr=r"""type of scaling to use. 'standard' uses the mean and standard deviation,
                    while 'robust' uses the median and interquartile range.""")
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._centerFunc = None
    self._scaleFunc = None

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['period'] = spec.parameterValues['period']
    settings['scalingType'] = spec.parameterValues.get('scaling', 'standard')
    self._centerFunc = np.median if settings['scalingType'] == 'robust' else np.mean
    self._scaleFunc = iqr if settings['scalingType'] == 'robust' else np.std
    return settings

  def fit(self, signal, pivot, targets, settings):
    """
      Fits the algorithm/model using the provided time series ("signal") using methods specific to
      the algorithm.
      @ In, signal, np.array, time-dependent series
      @ In, pivot, np.array, time-like parameter
      @ In, targets, list(str), names of targets
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, params, dict, characterization of signal; structure as:
                           params[target variable][characteristic] = value
    """
    params = {}
    period = settings['period']
    # FIXME This assumes there are no missing values in the signal. We should check the pivot values
    # instead of relying on the indexing of the values in signal.
    for tg, target in enumerate(targets):
      targetSignal = signal[:, tg].reshape(-1, period)
      params[target] = {'centers': self._centerFunc(targetSignal, axis=0),
                        'scales': self._scaleFunc(targetSignal, axis=0)}
    return params

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
    period = settings['period']
    residual = np.zeros_like(initial)
    # FIXME This assumes there are no missing values in the signal. We should check the pivot values
    # instead of relying on the indexing of the values in signal.
    for tg, (target, data) in enumerate(params.items()):
      # Reshape the data to be of shape (len(initial)//period, period)
      signal = initial[:, tg].reshape(-1, period)
      signal -= data['centers']
      signal /= data['scales']
      residual[:, tg] = signal.ravel()
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
    period = settings['period']
    composite = np.zeros_like(initial)
    # FIXME This assumes there are no missing values in the signal. We should check the pivot values
    # instead of relying on the indexing of the values in signal.
    for tg, (target, data) in enumerate(params.items()):
      # Reshape the data to be of shape (len(initial)//period, period)
      signal = initial[:, tg].reshape(-1, period)
      signal *= data['scales']
      signal += data['centers']
      composite[:, tg] = signal.ravel()
    return composite

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.fit
      @ Out, None
    """
    # Add model settings as subnodes to writeTO node
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      writeTo.append(base)


class PreserveCDF(TimeSeriesTransformer):
  """ Transformer that preserves the CDF of the input data """
  templateTransformer = skl.QuantileTransformer(output_distribution='uniform')

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'preservecdf'
    specs.description = r"""transforms the data to fit a given distribution by mapping the data to
    a uniform distribution and then to the desired distribution."""
    return specs

  def fit(self, signal, pivot, targets, settings):
    """
      Fits the algorithm/model using the provided time series ("signal") using methods specific to
      the algorithm.
      @ In, signal, np.array, time-dependent series
      @ In, pivot, np.array, time-like parameter
      @ In, targets, list(str), names of targets
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, params, dict, characterization of signal; structure as:
                           params[target variable][characteristic] = value
    """
    params = {}
    for tg, target in enumerate(targets):
      targetSignal = signal[:, tg].reshape(-1, 1)  # Reshape to be a column vector
      inputToUniform = deepcopy(self.templateTransformer).fit(targetSignal)
      params[target] = {'inputToUniform': inputToUniform}
    return params

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
    # Nothing to do on the forward transformation
    return initial

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
    composite = np.zeros_like(initial)
    for tg, (target, data) in enumerate(params.items()):
      # Reshape to column vector for scikit-learn transformer
      signal = initial[:, tg].reshape(-1, 1)
      # Transform from current distribution to a uniform distribution
      signal = deepcopy(self.templateTransformer).fit_transform(signal)
      # Now transform from a uniform distribution to the saved input distribution
      signal = data['inputToUniform'].inverse_transform(signal)
      # Flatten back to a row vector
      composite[:, tg] = signal.ravel()
    return composite

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.fit
      @ Out, None
    """
    # Add model settings as subnodes to writeTO node
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      writeTo.append(base)
