# Capability-Preference Decomposition for Adaptive Multi-Tenant LLM Routing

**Authors:**
- Sriram M
- Senthilvadivu K
- Sathya D
- Ramya G
- Sivaprakash M P
(Dept. of AI and Data Science, Kongu Engineering College, Perundurai, India)

## Abstract
Multi-tenant API gateways must balance quality, latency, and cost when routing queries across large language model (LLM) providers. Because these objectives vary across tenants and change over time, routing policies must adapt online. Standard contextual-bandit approaches handle adaptation by collapsing multidimensional provider outcomes into a scalar utility before updating their parameters. This design couples the provider’s estimated capability to whichever tenant preferences were active during training, forcing the system to relearn capabilities from scratch whenever a tenant’s priorities shift. We present Transfer-Learned LinUCB (TL-LinUCB), an algorithm that separates these quantities. TL-LinUCB estimates provider capability as a vector directly from raw outcome observations. Tenant preferences enter the decision purely as a linear projection at routing time. To address tenant cold-start, the algorithm blends a global ridge-regression estimator with a tenant-specific one, weighting them by their relative directional uncertainty. Evaluated across 30 random seeds against five synthetic provider profiles, this decomposition reduces cumulative pseudo-regret by 13.4% compared to a scalar monolithic baseline. When tenant preferences drift abruptly, TL-LinUCB cuts regret by 19.8% against a reset scalar policy because the preference shift requires only a new projection, not capability re-estimation. Finally, we empirically identify a boundary condition where exploration overhead erases the adaptive benefit, demonstrating when static routing remains the superior choice.

**Index Terms** — LLM routing, contextual bandits, multi-tenant systems, multi-objective optimization, online learning.

*(Note: This is the extracted text from the uploaded PDF document)*
