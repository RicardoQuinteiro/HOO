from hoo.experiments.fas import set_seed
from hoo.state_actions.hoo_state import HOOState
from hoo.hoot.hoot import HOOT
from hoo.environments.cartpole import ContinuousCartPole

set_seed()
D = 50
initial_state = HOOState(ContinuousCartPole())
hoot = HOOT(D, initial_state)
print(hoot.run(n=200))

set_seed()
hoot = HOOT(D, initial_state)
print(hoot.run(n=200))
