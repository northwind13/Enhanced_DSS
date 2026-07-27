=== MISSION BRIEF (standing; read once, applies to every decision of this incident) ===
MAP: 200 x 200 cells at 30 m; land cover: non fuel 5%, grass 18%, shrub 21%, pine litter 34%, hardwood 9%, water 12%, urban 1%.
- water: 4800 cells of lake/river/sea, centred near (158,38); engines and aircraft can draft/refill there, so sustained capacity near the water is real.
- assets on the map (protect by value and life):
  population 'City residents' at (150,184), ~31135 persons, value 1.0
  building 'City centre' at (150,184), value 1.0
  critical 'Hospital' at (149,186), value 1.0
  critical 'Power station' at (152,180), value 0.9
  population 'Village 1 residents' at (49,149), ~28865 persons, value 1.0
  building 'Village 1 centre' at (49,149), value 0.7
  evac_route 'Evacuation route' at (150,199), value 1.0
  evac_route 'Evacuation route 2' at (199,170), value 1.0
  evac_route 'Evacuation route 3' at (0,181), value 1.0
- resources: staged capacity 834 (rcap sum), map-mean travel time 49 min, air support: yes.
- protection priority weights (WHERE effort goes): critical 0.40, population 0.25, building 0.20, evacuation 0.15
- loss weights of the decision cost J (WHAT counts as a bad outcome): burn 1.00, asset 1.00, population 1.00, response 0.20, delay 0.20. Trials and cross-run comparisons are judged on the PHYSICAL part (burn + asset + population).
DOCTRINE FAMILIES (seed rules draw only on these): suppression_effort (suppression effort), resource_deployment (resource deployment), containment_line (containment line), asset_protection (asset protection), evacuation (evacuation), public_warning (public warning)
ACTUATOR LIBRARY (physics available beyond the doctrine; NO seed rule orders them, discovering a use is your job):
  tactical_burn: a firing crew ignites a strip between the containment band and the front so a counter-fire consumes the fuel the head fire is running toward. Real fire: in strong wind or near assets it can backfire, and the forecast gates will reject a reckless order.
  water_drafting: engines and helicopters refill from the nearest lake, river or sea, raising sustained capacity near water. On a map without water it does nothing.
  retardant_drop: aircraft coat the fuel ahead of the head fire with long-term retardant or soil, so the coated cells resist ignition even after the water in them dries. Aerial delivery: it does not need road access, but it is a finite, expensive pass and only the head sector is coated.
YOU MAY ALSO DEFINE A NEW ACTUATOR as data (a package with "clauses"): each clause = one verified effect (wet, clear, ignite, coat, evacuate, prime, draft) on a sector (head, flank, rear, ring, at_fire, assets, populated) at a cell range [rin, rout] from the front with an amount. That is a genuinely new tactic (WHERE and HOW); a mere re-weighting of channels is the tuning stage's job and fails the novelty gates.
WHEN TO INVENT WHAT (think like an incident commander):
  - a plain rule: the situation is expressible in the current concepts and an existing action answers it.
  - a new CONCEPT: the same kind of situation keeps recurring and the five decision concepts cannot name it (for example ember pressure on a settlement edge, urban interface stress). Concepts change the architecture: they enter Layer 3 and later rules can cite them.
  - a composite intervention: two channels must act AS ONE with a fixed ratio (hardening a town: protection + line).
  - a clause actuator: the tactic needs its own geometry (a flank firing operation, a deep pre-wetted band). Use only what THIS map supports: no water means no drafting and no aerial drops; no population nearby means evacuate and prime are wasted orders.
STANDING RULES OF THE GAME: every proposal must FIRE in the situation it is proposed for (use current dominant terms, or '>=' for a rising threat); orders must be strong enough to move a 45-minute physical forecast; every candidate is judged by simulation gates (form, vocabulary, relevance, availability, two reseeded A/B forecasts, and a growth margin for new vocabulary); a rejected proposal comes back to you once with the failing gate named, so fix exactly that. Always return ONLY the JSON.
=== END OF MISSION BRIEF ===
