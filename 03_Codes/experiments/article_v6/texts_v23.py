PARS = [
("Fig. 1 presents VODA as an SoS", """Fig. 1 presents the simulation core and its interfaces. VODA is an SoS with five kinds of participant. External providers supply weather, imagery, terrain, and resource status. A simulation core owns the hazard state. N local decision agents are each responsible for one region. A coordination layer allocates capacity without performing inference. An operator supervises through a dashboard. Participants meet only at declared interfaces. The simulation core is authoritative. Cells carry burning status, fuel mass, an intensity proxy, and time since ignition, and no other participant may write this state. The decision agents read derived observations. The dashboard admits only reversible, input-level actions such as pause, rollback, ignition injection, and resource changes."""),
("Fig. 1. VODA as an SoS", """Fig. 1. The simulation core. The external input set drives the grid state through the hybrid transition; the decision core reads the state only through the observation function and acts only through the intervention field; snapshots with the random state enable exact counterfactual replay."""),
("Fig. 2. One local agent", """Fig. 2. The decision core of one local agent. Heterogeneous inputs become ten confidence-rated features, six base concepts, and five decision concepts; the closed layer carries the self-tuning TS doctrine with the satisficing and quality gates; the open layer extends the doctrine through gated proposals; the coordinator shares the pool across agents."""),
]
INSERT_AFTER = []
DELETE = []
IMAGES = {1: "fig1_simcore", 2: "fig2_decisioncore"}
CX = {1: 3017520, 2: 3017520}
