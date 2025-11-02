from src import APITOKEN
NIRAJ_TOKEN = APITOKEN.NIRAJ_TOKEN

from dwave.system import LeapHybridCQMSampler
import dimod

# Create model
cqm = dimod.ConstrainedQuadraticModel()
cqm.add_variable(dimod.BINARY, "x")
x = dimod.Binary("x")
cqm.set_objective(x)

# Solve using D-Wave hybrid CQM solver
sampler = LeapHybridCQMSampler(
    token=NIRAJ_TOKEN,
    endpoint="https://cloud.dwavesys.com/sapi",
    solver="hybrid_constrained_quadratic_model_version1p"
)

result = sampler.sample_cqm(cqm)

# Output result
print("Sample:", result.first.sample)
print("Objective value:", result.first.energy)
