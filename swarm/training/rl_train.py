"""
Placeholder for MARL / supervised training routines.
The stub below simply echoes that training is not wired yet but
keeps imports from failing.
"""


def train(dataset, policy_cls, **kwargs):
    # TODO: integrate with your preferred deep learning framework.
    return {
        "status": "not_implemented",
        "dataset_size": len(getattr(dataset, "buffer", [])),
        "policy": getattr(policy_cls, "__name__", str(policy_cls)),
    }
