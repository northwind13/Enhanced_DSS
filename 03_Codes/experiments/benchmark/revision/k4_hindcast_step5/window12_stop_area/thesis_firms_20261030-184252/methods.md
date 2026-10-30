**How the validation works.** Each case is a blind hindcast: the simulator is given only the real inputs of the documented fire (Copernicus GLO-30 terrain, ESA WorldCover fuel, hourly ERA5 weather at the ignition point, and the first NASA FIRMS overpass as the initial front) and is run for the recorded duration with no tuning on that fire.

The simulation core models **free fire spread; it does not model suppression**, while the real fire was actively fought. A free run therefore overpredicts the FINAL burned area, so the classic area-overlap scores (Sorensen-Dice, Jaccard, false-alarm ratio, area bias) are NOT used: they would conflate the spread error with the missing suppression and with the satellite undersampling. The model is validated on what it actually claims, the **propagation of the fire front**, with three criteria ($A$ = simulated burn, $B$ = observed burn):

| Criterion | Definition | What it checks |
|---|---|---|
| Coverage (POD) | $|A\cap B|/|B|$ | fraction of the observed burn the model reproduced. Robust to the free-run overshoot. Target $>0.7$. |
| Front position error | mean and 90th-percentile edge-to-edge distance between $\partial A$ and $\partial B$ (m) | how far the simulated front sits from the observed front |
| Arrival-time agreement | over cells both burned, mean absolute difference between the simulated arrival time and the FIRMS first-detection time (h), and the Spearman rank correlation $\rho$ of the arrival ORDER | the rate-of-spread test: does the front reach each place at about the right time, in the right order |

Ember spotting is stochastic, so every case is run over several seeds and each criterion is reported as mean +/- standard deviation. Area-overlap scores are reported only when an official EFFIS/EMS perimeter supplies referee-grade area truth.