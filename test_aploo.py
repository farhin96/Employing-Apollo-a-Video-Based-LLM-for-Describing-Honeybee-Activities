import os, sys
print("CWD:", os.getcwd())  # should be C:\Users\Windows\aploo

# make sure project root is on the path (usually already is)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import utils.mm_utils as mm
import utils.conversation as conv
import utils.constants as const

print("mm_utils loaded from:", mm.__file__)
print("conversation loaded from:", conv.__file__)
print("constants loaded from:", const.__file__)
