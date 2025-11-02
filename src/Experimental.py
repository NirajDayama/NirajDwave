#D-Wave token
##   h2DI-aaa26ce9ed5fb1f2c3e2af570f49106abddb76de

import APITOKEN
nirajToken = APITOKEN.NIRAJ_TOKEN

from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model
import sys


from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model

sampler = LeapHybridNLSampler(token=nirajToken, region="na-west-1")

model = Model()
x = model.integer(lower_bound=3, upper_bound=6 )
model.minimize(x**2)


# ✅ Export using model.serialize()
model.into_file("model_save.json")

sampleset = sampler.sample(model, label='Niraj-Test-Problem')
result = sampleset.result()
print("Sampleset is: ",sampleset)
print("Result is:",result)

decision_iterator = model.iter_decisions()

try:
    # Read the symbols one by one from the iterator

    # Read Variable i (Index 0)
    i_symbol = next(decision_iterator)
    # FIX: Use the '.state' attribute to access the solved numerical value.
    print(f"i (integer) = {i_symbol.state()}")


except StopIteration:
    print("Error: The decision iterator ran out of items before all variables were read.")
    sys.exit(1)
except AttributeError as e:
    # If .state fails, this is the final confirmation of a serious version bug.
    print(f"CRITICAL ERROR: Failed to read attribute '.state'. Details: {e}")
    print("This confirms a version bug. Please run: pip install --upgrade dwave-optimization dwave-system")
    sys.exit(1)


sys.exit(1)


# Step 1: Create the sampler
sampler = LeapHybridNLSampler(
    region="na-west-1",
    token=nirajToken,
)

print("✅ Created sampler:", sampler.solver)

# Step 2: Build the model
model = Model()
print("✅ Model created")

# Step 3: Define variables
i = model.integer(lower_bound=-5, upper_bound=5)
c = model.constant(4)
x = model.integer(lower_bound=0, upper_bound=5)
z = model.binary()
print("✅ Variables defined")

# Step 4: Define objective
y = i**2 - c * i
model.minimize(y)
print("✅ Objective set")

# Step 5: Add constraint
model.add_constraint(x + i + z <= c)
print("✅ Constraint added")

# Step 6: Submit the model
print("🚀 Submitting model...")
result = sampler.sample(model).result()

# Step 7: Inspect result structure
print("\n📦 Raw result:")
print(result)


# Step 4: Access info safely
info = result.info
timing = info.get("timing", {})
warnings = info.get("warnings", [])

print("\n⏱ Timing:")
for key, val in timing.items():
    print(f"{key}: {val}")

if warnings:
    print("\n⚠️ Warnings:")
    for w in warnings:
        print("-", w)
else:
    print("\n✅ No warnings reported.")

# Step 5: Check for solution
if hasattr(result, "solutions") and result.solutions:
    best = result.solutions[0]
    print("\n✅ Best Solution:")
    for var, val in best.values.items():
        print(f"{var} = {val}")
    print(f"Objective value: {best.objective}")
else:
    print("\n⚠️ No solution returned. The model may be infeasible or disconnected.")


sys.exit(1)


from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model
import sys
import os



sampler = LeapHybridNLSampler(
    region="na-west-1",
    token=nirajToken,
    # solver="hybrid_nonlinear_program_version1p"
)
print("I have created ",sampler.solver, ". We are making model now.")

model = Model()

print("We made model as ",model,". We are making variables now.")

# Define variables
i = model.integer(lower_bound=-5, upper_bound=5)
c = model.constant(4)
x = model.integer(lower_bound=0, upper_bound=5)
z = model.binary()

print("We defined variables. We are setting obj and constraints now.")

# Define expression
y = i**2 - c*i

# Set objective
model.minimize(y)

model.add_constraint(x + i + z <= c)

print("Model is ready. We are executing now.")

future = sampler.sample(model)
sampling_result = future.result()
print("Sampling done. Sampling result is: ",sampling_result)

print("Now we will read model.iter_decisions()")

try:
    best_decision = next(model.iter_decisions())
    print("best_decision is:",best_decision)
except StopIteration:
    print("Error: Solver failed to find any feasible solution.")
    sys.exit(1)

# Access best sample and objective
# Step 4: Access solution
solution = sampling_result.solution
values = solution.values
objective = solution.objective

# Step 5: Print results
print("\n✅ Best Solution:")
for var, val in values.items():
    print(f"{var} = {val}")
print(f"Objective value: {objective}")



# Display results
print("\n🔍 best_Sample: ",best_sample, "\n And it gives variables values as : \n")
for var, val in best_sample.items():
    print(f"{var} = {val}")
print(f"Objective value: {objective}")





print("The best_decision is: ",best_decision)

# Display solution
print("\n🔍 Solution:")

# ... (Code leading up to retrieving best_decision remains the same)

# FIX 1: Use iter_decisions() to retrieve the best solution state (decision set)
# We assume the first decision is the best and retrieve it with next().
try:
    best_decision = next(model.iter_decisions())
except StopIteration:
    print("Error: Solver failed to find any feasible solution.")
    sys.exit(1)


# Display solution
print("\n🔍 Solution:")


# FIX 2: Access the solved value using the get_value() method
print(f"i (integer) = {best_decision.get_value(i)}")
print(f"x (integer) = {best_decision.get_value(x)}")
print(f"z (binary)  = {best_decision.get_value(z)}")

# Access the constant value (constants are symbols, but don't change)
print(f"c (constant) = {c.value}")

# Access the calculated objective value (this remains correct)
print(f"Objective value: {model.objective.value}")



print(f"i (integer) = {best_decision[i]}")
print(f"x (integer) = {best_decision[x]}")
print(f"z (binary)  = {best_decision[z]}")

# Access the constant value (constants are symbols, but don't change)
print(f"c (constant) = {c.value}") # The constant symbol *does* have a .value property

# Access the calculated objective value (this is correct)
print(f"Objective value: {model.objective.value}")

# Optional: Display timing info (using the result_metadata variable)
print("\n⏱ Timing:")
# Note: The timing is nested under the 'timing' key in the metadata dictionary
for key, val in result_metadata['timing'].items():
    print(f"{key}: {val}")
print(f"Problem ID: {result_metadata['problem_id']}")


print("-------------------DONE----------------")




# FIX 1: Capture the result into a single variable, as it's the metadata/timing.
# The actual solution values are written back to the 'model' object.
result_metadata = sampler.sample(model).result()

# The solution is now implicitly available in the model's symbols (i, x, z)
# and the objective (y).

print("We executed OK. We got result metadata as follows:")

# FIX 2: Rename the print variable for clarity
print("Result Metadata object is: ", result_metadata)
print("The Model object is still: ", model)
print("Now we will try to read values directly from the model's symbols.")

# FIX 3: Access values directly from the symbols created earlier.

# Display solution
print("\n🔍 Solution:")

# Access individual variable values
print(f"i (integer) = {i.solution}")
print(f"x (integer) = {x.solution}")
print(f"z (binary)  = {z.solution}")




# Access the constant value (should remain 4)
print(f"c (constant) = {c.value}")

# Access the calculated objective value
print(f"Objective value (y = i^2 - c*i): {y.value}")

# Optional: Display timing info (using the result_metadata variable)
print("\n⏱ Timing:")
for key, val in result_metadata['timing'].items():
    print(f"{key}: {val}")
print(f"Problem ID: {result_metadata['problem_id']}")


#### result = sampler.sample(model).result()





##############################################

solution, timing = sampler.sample(model).result()

print("We executed OK. We got solution and timings as follows:")

print("Solution object is: ",solution)
print("Timing object is ",timing)
print("Now we will try to read values from inside the solution object")

# Display solution
print("\n🔍 Solution:")
for var, val in solution.values.items():
    print(f"{var} = {val}")
print(f"Objective value: {solution.objective}")

# Optional: Display timing info
print("\n⏱ Timing:")
for key, val in timing.items():
    print(f"{key}: {val}")
