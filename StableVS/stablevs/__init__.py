"""
StableVS: Stable Video Synthesis Scheduler Extensions

This package provides custom schedulers for diffusers that implement
fast-low sampling strategies for improved generation quality.
"""

__version__ = "0.1.0"

from .schedulers.scheduling_stablevs_dpmsolver_multistep import StableVSDPMSolverMultistepScheduler
from .schedulers.scheduling_stablevs_flow_match import StableVSFlowMatchScheduler, FlowMatchEulerDiscreteSchedulerOutput
from .schedulers.scheduling_stablevs_unipc_multistep import StableVSUniPCMultistepScheduler

__all__ = [
    "StableVSDPMSolverMultistepScheduler",
    "StableVSFlowMatchScheduler",
    "FlowMatchEulerDiscreteSchedulerOutput",
    "StableVSUniPCMultistepScheduler",
]
