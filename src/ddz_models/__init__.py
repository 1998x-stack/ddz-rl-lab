"""Role-aware candidate-action Q networks (PyTorch optional extra)."""
from .network import CandidateQNetwork, NetworkSpec
from .policy import RoleBundle, role_for_observation
__all__ = ["CandidateQNetwork", "NetworkSpec", "RoleBundle", "role_for_observation"]
