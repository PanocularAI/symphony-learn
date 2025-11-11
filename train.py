import sys
import runpy

# Import custom train_spec
import train_spec

# Inject the module into sys.modules under the name train.py expects
sys.modules["torchtitan.protocols.train_spec"] = train_spec

runpy.run_module("torchtitan.train", run_name="__main__")