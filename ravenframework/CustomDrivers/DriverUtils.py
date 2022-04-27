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
  Utilities common to Drivers for RAVEN.
"""

import os
import sys

# ***********************************************
# main utilities
#
def doSetup():
  """
    Fully sets up RAVEN environment and variables.
    @ In, None
    @ Out, None
  """
  printStatement()
  printLogo()
  setupBuiltins()
  setupWarnings()
  setupFramework()
  setupH5py()
  setupCpp()
  checkVersions()

def findFramework():
  """
    Provides path to framework dir
    @ In, None
    @ Out, findFramework, str, framework dir
  """
  return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ***********************************************
# a la carte setup
#
def setupBuiltins():
  """
    Sets up the memory tracking profile tool, or bypasses it if not available.
    @ In, None
    @ Out, None
  """
  import builtins
  try:
    builtins.profile
  except (AttributeError,ImportError):
    # profiler not preset, so pass through
    builtins.profile = lambda f: f

def setupWarnings():
  """
    sets up warnings status
    @ In, None
    @ Out, None
  """
  import warnings
  if not __debug__:
    warnings.filterwarnings("ignore")
  else:
    warnings.simplefilter("default", DeprecationWarning)

def setupFramework():
  """
    sets up framework on path
    @ In, None
    @ Out, None
  """
  frameworkDir = findFramework()
  if frameworkDir not in sys.path:
    sys.path.append(frameworkDir)

def setupH5py():
  """
    sets up env vars for h5py
    @ In, None
    @ Out, None
  """
  #warning: this needs to be before importing h5py
  os.environ["MV2_ENABLE_AFFINITY"]="0"

def setupCpp():
  """
    Find and add c++ libs to path
    @ In, None
    @ Out, None
  """
  frameworkDir = findFramework()

  from ravenframework.utils import utils
  utils.find_crow(frameworkDir)

  if any(os.path.normcase(sp) == os.path.join(frameworkDir,'contrib') for sp in sys.path):
    print(f'WARNING: "{os.path.join(frameworkDir,"contrib")}" already in system path. Skipping CPP setup')
  else:
    utils.add_path(os.path.join(frameworkDir,'contrib'))
    ##TODO REMOVE PP3 WHEN RAY IS AVAILABLE FOR WINDOWS
    utils.add_path_recursively(os.path.join(frameworkDir,'contrib','pp'))

def checkVersions():
  """
    Method to check if versions of modules are new enough. Will call sys.exit
    if they are not in the range specified.
    @ In, None
    @ Out, None
  """
  # import library handler
  frameworkDir = findFramework()
  scriptDir = os.path.join(frameworkDir, '..', 'scripts')
  if scriptDir not in sys.path:
    remove = True
    sys.path.append(scriptDir)
  else:
    remove = False
  import library_handler as LH
  if remove:
    sys.path.pop(sys.path.index(scriptDir))
  # if libraries are not to be checked, we're done here
  if not LH.checkVersions():
    return
  # otherwise, we check for incorrect libraries
  missing, notQA = LH.checkLibraries()
  if missing:
    print('ERROR: Some required Python libraries are missing but required to run RAVEN as configured:')
    for lib, version in missing:
      # report the missing library
      msg = '  -> MISSING: {}'.format(lib)
      # add the required version if applicable
      if version is not None:
        msg += ' version {}'.format(version)
      print(msg)
  if notQA:
    print('ERROR: Some required Python libraries have incorrect versions for running RAVEN as configured:')
    for lib, found, need in notQA:
      print('  -> WRONG VERSION: lib "{}" need "{}" but found "{}"'.format(lib, found, need))
  if missing or notQA:
    print('Try installing libraries using instructions on RAVEN repository wiki at ' +
           'https://github.com/idaholab/raven/wiki/Installing_RAVEN_Libraries.')
    sys.exit(-4)
  else:
    print('RAVEN Python dependencies located and checked.')
  # TODO give a warning for missing libs even if skipping check?
  # -> this is slow, so maybe not.

def printStatement():
  """
    Method to print the BEA header
    @ In, None
    @ Out, None
  """
  print("""
Copyright 2017 Battelle Energy Alliance, LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
  """)

def printLogo():
  """
    Method to print a RAVEN logo
    @ In, None
    @ Out, None
  """
  print("""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      .---.        .------######       #####     ###   ###  ########  ###    ###
     /     \  __  /    --###  ###    ###  ###   ###   ###  ###       #####  ###
    / /     \(  )/    --###  ###    ###   ###  ###   ###  ######    ### ######
   //////   ' \/ `   --#######     #########  ###   ###  ###       ###  #####
  //// / // :    :   -###   ###   ###   ###    ######   ####      ###   ####
 // /   /  /`    '---###    ###  ###   ###      ###    ########  ###    ###
//          //..\\
===========UU====UU=============================================================
           '//||\\`
             ''``
    """)