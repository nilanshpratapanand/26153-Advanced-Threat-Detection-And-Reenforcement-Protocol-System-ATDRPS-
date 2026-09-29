"""Attack-onset forecasting: an evaluation protocol, baselines and a hazard model.

The v1 pipeline scored "is infiltration active in the next window", which a model can
answer well simply by noticing an attack that is already under way.  This package asks
the harder, useful question -- *given that nothing has happened recently, will an
infiltration begin within the next K windows?* -- and measures it the way the security
ML literature says it must be measured (see docs/RESEARCH_AND_PLAN_V2.md).
"""
