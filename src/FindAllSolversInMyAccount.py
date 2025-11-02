from dwave.cloud import Client
import APITOKEN
NIRAJ_TOKEN = APITOKEN.NIRAJ_TOKEN

with Client(token=NIRAJ_TOKEN, endpoint="https://cloud.dwavesys.com/sapi") as client:
    solvers = client.get_solvers()
    for name in solvers:
        print(name)
