# Standard Library Imports
import sys

# Hermes Imports
import hermes
import hermes.configuration
import hermes.system

# The reference pickles in tests/test_data were written by pyzentropy and name their classes by
# its module paths. Point those names at the Hermes modules so the files load unchanged.
sys.modules.setdefault("pyzentropy", hermes)
sys.modules.setdefault("pyzentropy.configuration", hermes.configuration)
sys.modules.setdefault("pyzentropy.system", hermes.system)
