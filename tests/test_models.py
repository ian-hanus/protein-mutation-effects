import numpy as np
import torch

from pme.models import MutationGNN, build_adjacency

L, D = 12, 16
COORDS = np.stack([np.arange(L) * 5.0, np.zeros(L), np.zeros(L)], axis=1).astype(np.float32)


def _predict(model, node_x, adj, pos=3):
    model.eval()
    p = torch.tensor([pos])
    return model(node_x, adj, p, torch.tensor([0]), torch.tensor([1]), torch.tensor([0.5]))


def test_adjacency_kinds():
    ident = build_adjacency("identity", COORDS)
    assert torch.allclose(ident, torch.eye(L))
    window = build_adjacency("window", COORDS, window=2)
    assert window[0, 2] > 0 and window[0, 3] == 0
    contact = build_adjacency("contact", COORDS, cutoff=8.0)  # 5 A apart -> only direct neighbours
    assert contact[3, 4] > 0 and contact[3, 5] == 0


def test_identity_graph_ignores_other_residues():
    torch.manual_seed(0)
    model = MutationGNN(D, use_llr=True)
    x = torch.randn(L, D)
    x2 = x.clone()
    x2[7] += 10 * torch.randn(D)  # perturb a residue that is not the mutated one
    a = build_adjacency("identity", COORDS)
    assert torch.allclose(_predict(model, x, a), _predict(model, x2, a), atol=1e-6)


def test_contact_graph_propagates_neighbour_information():
    torch.manual_seed(0)
    model = MutationGNN(D, use_llr=True)
    x = torch.randn(L, D)
    x_near, x_far = x.clone(), x.clone()
    x_near[4] += 10 * torch.randn(D)  # contact of residue 3
    x_far[9] += 10 * torch.randn(D)   # far from residue 3
    a = build_adjacency("contact", COORDS)
    base = _predict(model, x, a)
    assert not torch.allclose(base, _predict(model, x_near, a), atol=1e-4)
    assert torch.allclose(base, _predict(model, x_far, a), atol=1e-6)


def test_gradients_flow():
    model = MutationGNN(D, use_llr=False)
    x = torch.randn(L, D)
    out = model(x, build_adjacency("window", COORDS), torch.tensor([1, 2]), torch.tensor([0, 1]), torch.tensor([2, 3]))
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
