from src import APITOKEN
NIRAJ_TOKEN = APITOKEN.NIRAJ_TOKEN

from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model
import sys

sampler = LeapHybridNLSampler(token=NIRAJ_TOKEN, region="na-west-1")
model = Model()

x = model.integer(lower_bound=2, upper_bound=5 )
y = model.integer(lower_bound=2, upper_bound=5 )
model.minimize(x**2 - y**2)

# ✅ Export using model.serialize()
model.into_file("Minimal_Model_Saved")

sampleset = sampler.sample(model, label='Niraj-Minimal-NL-TestProblem')
result = sampleset.result()
print("Sampleset is: ",sampleset)
print("Result is:",result)
decision_iterator = model.iter_decisions()

try:
    # Read the symbols one by one from the iterator

    # Read Variable x (Index 0)
    x_symbol = next(decision_iterator)
    print(f"x (value should be min 2) = {x_symbol.state()}")

    # Read Variable x (Index 1)
    y_symbol = next(decision_iterator)
    print(f"y (value should be max 5) = {y_symbol.state()}")

except StopIteration:
    print("Error: The decision iterator ran out of items before all variables were read.")
    sys.exit(1)
except AttributeError as e:
    # If .state fails, this is the final confirmation of a serious version bug.
    print(f"CRITICAL ERROR: Failed to read attribute '.state'. Details: {e}")
    print("This confirms a version bug. Please run: pip install --upgrade dwave-optimization dwave-system")