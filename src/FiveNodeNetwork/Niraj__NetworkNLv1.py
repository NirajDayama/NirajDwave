from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model
import sys

from src import APITOKEN
NIRAJ_TOKEN = APITOKEN.NIRAJ_TOKEN

sampler = LeapHybridNLSampler(token=NIRAJ_TOKEN, region="na-west-1")

# 📍 Network setup
nodes = [1, 2, 3, 4, 5]
sources = [1, 2]
sinks = [4, 5]
S_init = 10
S_max = 20
G_max = {1: 15, 2: 10}
D = {4: 8, 5: 12}
C_gen = {1: 2.0, 2: 3.5}

# 🧠 Build symbolic model
model = Model()

# 🚚 Flow variables
x = {}
for i in nodes:
    for j in nodes:
        if i != j:
            x[i, j] = model.integer(lower_bound=0)

# ⚡ Generation variables
g = {}
for i in sources:
    g[i] = model.integer(lower_bound=0, upper_bound=G_max[i])

# 🧊 Storage variable
s = model.integer(lower_bound=0, upper_bound=S_max)

# 🎯 Objective: Minimize generation cost
model.minimize(C_gen[1]*g[1] + C_gen[2]*g[2])

# ⚖️ Source node balance
for i in sources:
    outflow = sum(x[i, j] for j in nodes if j != i)
    inflow = sum(x[j, i] for j in nodes if j != i)
    model.add_constraint(outflow - inflow == g[i])

# ⚖️ Sink node balance
for i in sinks:
    inflow = sum(x[j, i] for j in nodes if j != i)
    outflow = sum(x[i, j] for j in nodes if j != i)
    model.add_constraint(inflow - outflow == D[i])

# ⚖️ Storage node balance (Node 3)
inflow = sum(x[j, 3] for j in nodes if j != 3)
outflow = sum(x[3, j] for j in nodes if j != 3)
model.add_constraint(inflow - outflow == s - S_init)

# 💾 Save model
model.into_file("Niraj_5Node_Model")

# 🚀 Solve with LeapHybridNLSampler
sampleset = sampler.sample(model, label='Niraj-5Node-NetworkFlow')
result = sampleset.result()

# 📤 Extract results
decision_iterator = model.iter_decisions()
try:
    decision_iterator = model.iter_decisions()

    print("Flow Variables:")
    for i in nodes:
        for j in nodes:
            if i != j:
                print(f"x_{i}_{j} = {next(decision_iterator).state()}")

    print("\nGeneration Variables:")
    print(f"g_1 = {next(decision_iterator).state()}")
    print(f"g_2 = {next(decision_iterator).state()}")

    print("\nStorage Variable:")
    print(f"s = {next(decision_iterator).state()}")

except StopIteration:
    print("❌ Error: Decision iterator ran out of items.")
    sys.exit(1)
except AttributeError as e:
    print(f"❌ CRITICAL ERROR: {e}")
    print("Please run: pip install --upgrade dwave-optimization dwave-system")
