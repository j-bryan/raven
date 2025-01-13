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
Created on November 13, 2023
@author: sotogj

Filter Bank Discrete Wavelet Transform
"""

import numpy as np
import copy

from ..TimeSeriesAnalyzer import TimeSeriesTransformer
from ...utils import xmlUtils, InputTypes, InputData, importerUtils
try:
  pywt = importerUtils.importModuleLazy("pywt", globals())
except ModuleNotFoundError as exc:
  print("The FilterBankDWT TSA Module requires the PYWAVELETS library to be installed in the current python environment")
  raise ModuleNotFoundError from exc

class FilterBankDWT(TimeSeriesTransformer):
  """ Applies a Discrete Wavelet Transform algorithm as a filter bank to decompose signal into
      multiple time resolution signals.
  """
  _acceptsMissingValues = True
  _multiResolution = True

  def __init__(self, *args, **kwargs):
    """
      A constructor that will appropriately intialize a time-series analysis object
      @ In, args, list, an arbitrary list of positional values
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    # general infrastructure
    super().__init__(*args, **kwargs)
    # total number of decomposition levels, including approximation level and detail levels.
    # NOTE: the number of levels will start with user input but may be modified to lower level
    #       if algorithm determines it cannot decompose the signal further.
    self._levels = 1

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'filterbankdwt'
    specs.description = r"""Filter Bank Discrete Wavelet Transform, a multi-resolution-capable TimeSeriesAnalysis
        algorithm. Performs a discrete wavelet transform (DWT) on time-dependent data as a filter bank to decompose the
        signal to multiple frequency levels. Given a wavelet family and the original signal, the signal is projected
        onto modifications of the original mother wavelet $\Psi$ to produce wavelet coefficients. The modifications
        $\psi_{a,b}$ happen in two ways:
        \\
        \begin{itemize}
          \item the wavelet is scaled by factor $a$ to capture features at different time scales (e.g., if the wavelet
          is thinner it better captures faster frequency features)
          \item the wavelet is shifted in time by factor $b$ across the entire time domain of the signal for each
          scale $a$
        \end{itemize}
        After all projections, there is a 2-D array of coefficients regarding the scale $a$ and shift $b$. The modified
        wavelets are given by:
        \\
        \begin{equation*}
           \psi_{a,b} = \frac{1}{\sqrt{a}} \Psi(\frac{t-b}{a})
        \end{equation*}
        The Filter Bank DWT works in a cascading sequence of low- and high-pass filters for all requested decomposition
        levels to create the wavelet coefficients. The low- and high-pass filters create a set of approximation and
        detail coefficients, respectively, for each scale. Approximation coefficients correspond to lower
        frequency/large wavelength features; detail cofficients, to higher frequency/smaller wavelength features.
        Subsequent decompositions apply the filters to the previous approximation coefficients. For N levels of
        decomposition, N sets of detail coefficients and 1 set of approximation coefficients are produced. Currently,
        the approximation coefficients are treated as a trend in the signal and subtracted from the signal.
        Note: This TSA module requires pywavelets to be installed within your python environment."""
    specs.addSub(InputData.parameterInputFactory(
              'family',
              contentType=InputTypes.StringType,
              descr=r"""The type of wavelet to use for the transformation.
                    There are several possible families to choose from, and most families contain
                    more than one variation. For more information regarding the wavelet families,
                    refer to the Pywavelets documentation located at:
                    https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html (wavelet-families)
                    \\
                    Possible values are:
                    \begin{itemize}
                      \item \textbf{haar family}: haar
                      \item \textbf{db family}: db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11,
                        db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23,
                        db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35,
                        db36, db37, db38
                      \item \textbf{sym family}: sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10,
                        sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20
                      \item \textbf{coif family}: coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8,
                        coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17
                      \item \textbf{bior family}: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6,
                        bior2.8, bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5,
                        bior6.8
                      \item \textbf{rbio family}: rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6,
                        rbio2.8, rbio3.1, rbio3.3, rbio3.5, rbio3.7, rbio3.9, rbio4.4, rbio5.5,
                        rbio6.8
                      \item \textbf{dmey family}: dmey
                      \item \textbf{gaus family}: gaus1, gaus2, gaus3, gaus4, gaus5, gaus6, gaus7, gaus8
                      \item \textbf{mexh family}: mexh
                      \item \textbf{morl family}: morl
                      \item \textbf{cgau family}: cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8
                      \item \textbf{shan family}: shan
                      \item \textbf{fbsp family}: fbsp
                      \item \textbf{cmor family}: cmor
                    \end{itemize}"""))
    specs.addSub(InputData.parameterInputFactory('levels', contentType=InputTypes.IntegerType,
              descr=r"""the number of wavelet decomposition levels for requested for the signal. This is equivalent to
                    the number of sets of detail coefficients produced. Note that there will always be one set of
                    approximation cofficients produced, which is treated as a trend in the signal. Note that there is a
                    maximum decomposition level depending on signal length and the chosen wavelet family: if desired
                    level is larger than the maximum decomposition level, the latter will be used. Provided level must
                    be nonzero."""))
    return specs

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['family'] = spec.findFirst('family').value
    settings['levels'] = spec.findFirst('levels').value

    if settings['levels'] == 0:
      raise IOError("Discrete Wavelet Transform requires non-zero number of decomposition levels.")

    self._levels = settings['levels']

    return settings

  def fit(self, signal, pivot, targets, settings, trainedParams=None):
    """
      This function utilizes the Discrete Wavelet Transform to
      characterize a time-dependent series of data.

      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, additional settings specific to this algorithm
      @ In, trainedParams, dict, running dict of trained algorithm params
      @ Out, params, dict, characteristic parameters
    """
    # TODO extend to continuous wavelet transform

    ## The pivot input parameter isn't used explicity in the
    ## transformation as it assumed/required that each element in the
    ## time-dependent series is independent, uniquely indexed and
    ## sorted in time.
    family = settings['family']
    params = {target: {'results': {}} for target in targets}

    # determine maximum decomposition level
    max_level = pywt.dwt_max_level(len(pivot), family)
    if self._levels>max_level:
      print(f"Number of levels requested is larger than maximum DWT decomposition level, switching to maximum allowed: {max_level}")
      self._levels = max_level
      settings['levels'] = self._levels

    for i, target in enumerate(targets):
      history = signal[:, i]
      mask = np.isnan(history)
      history[mask] = 0
      # TODO:this is temporary for zero-filter SOLAR data... should this also look back to find filter results?

      results = params[target]['results']
      coeffs = pywt.mra(history, family, self._levels, transform='dwt')
      for coeff in coeffs:
        coeff[mask] = np.nan

      results['coeff_a'] = coeffs[0]
      results['coeff_d'] = np.vstack([coeffs[i] for i in range(1,self._levels+1)]) if self._levels>1 else coeffs[1][np.newaxis]

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
    residual = initial.copy()
    for i, target in enumerate(settings['target']):
      residual = initial[:,i] - params[target]['results']['coeff_a']
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
    synthetic = np.zeros((len(pivot), len(params)))
    for t, (target, _) in enumerate(params.items()):
      results = params[target]['results']
      cA = results['coeff_a']
      cD = results['coeff_d']
      synthetic[:, t] = pywt.imra(np.vstack([cA,cD]))
    composite = initial + synthetic
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
      if 'coeff_a' in info['results']:
        base.append(xmlUtils.newNode('N_approx_levels', text=1)) # number of approximation levels (should be 1)
      if 'coeff_d' in info['results']:
        coeff_d = info['results']['coeff_d']
        base.append(xmlUtils.newNode('N_detail_levels', text=len(coeff_d))) # number of detail levels (>1)

  def getDecompositionLevels(self):
    """
      Returns number of decomposition levels requested from user (overwritten by max. allowed
      per wavelet family as a function of signal length).
      @ In, None
      @ Out, levels, int, number of decomposition levels
    """
    return self._levels

  def sortTrainedParamsByLevels(self, params):
    """
      Sorts or reformats the training parameters dictionary in a manner specific to each
      multi resolution algorithm.
      @ In, params, dict, training parameters as from self.characterize
      @ Out, sortedParams, dict, reformatted training parameters
    """
    # reformatting the results of the trained `params` to be:
    #     {target: {lvl: [ values, ... ], }, }
    # this might look different per algorithm
    sortedParams = {}
    for target, contents in params.items():
      sortedParams[target] = {}
      for lvl in range(self._levels):
        sortedParams[target][lvl] = contents['results']['coeff_d'][lvl,:]
    return sortedParams

  def combineTrainedParamsByLevels(self, params, newParams):
    """
      Updates training parameter dictionary with trained parameters from previous
      decomposition levels.
      @ In, params, dict, original training parameters as from self.characterize
      @ In, newParams, dict, new training parameters from other decomposition levels
      @ Out, None
    """
    # reformatting the results of the trained `params` to fit this algo's format:
    #     {target: {lvl: [ values, ... ], }, }
    # this might look different per algorithm
    for target, originalContents in params.items():
      for lvl, newContents in newParams[target].items():
        originalContents['results']['coeff_d'][lvl,:] = newContents