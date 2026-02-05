import torch


def test_content_addressing_weights_shape_and_normalization():
    from exercise import ContentAddressing

    torch.manual_seed(0)
    B, N, K = 3, 7, 5
    memory = torch.randn(B, N, K)
    key = torch.randn(B, K)
    beta = torch.ones(B, 1)

    ca = ContentAddressing(memory_size=N, key_size=K)
    w = ca(key=key, beta=beta, memory=memory)
    assert w.shape == (B, N)
    assert torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-6)
    assert torch.all(w >= 0)


def test_content_addressing_beta_sharpens_distribution():
    from exercise import ContentAddressing

    torch.manual_seed(0)
    B, N, K = 2, 9, 4
    memory = torch.randn(B, N, K)
    key = memory[:, 3, :]  # make one location an exact match

    ca = ContentAddressing(memory_size=N, key_size=K)
    w_soft = ca(key=key, beta=torch.ones(B, 1) * 0.5, memory=memory)
    w_sharp = ca(key=key, beta=torch.ones(B, 1) * 5.0, memory=memory)
    assert torch.max(w_sharp, dim=1).values.mean() > torch.max(w_soft, dim=1).values.mean()


def test_location_addressing_shift_and_sharpen():
    from exercise import LocationAddressing

    torch.manual_seed(0)
    B, N = 2, 8
    la = LocationAddressing(memory_size=N, shift_range=1)

    prev = torch.zeros(B, N)
    prev[:, 2] = 1.0
    content = prev.clone()
    gate = torch.ones(B, 1)  # take content
    shift = torch.zeros(B, 3)
    shift[:, 2] = 1.0  # shift +1 ([-1,0,+1])
    gamma = torch.ones(B, 1) * 1.0

    w = la(content_weights=content, prev_weights=prev, gate=gate, shift=shift, gamma=gamma)
    expected = torch.zeros(B, N)
    expected[:, 3] = 1.0
    assert torch.allclose(w, expected, atol=1e-6)


def test_differentiable_stack_push_read_pop():
    from exercise import DifferentiableStack

    torch.manual_seed(0)
    B, D, E = 2, 5, 3
    stack = DifferentiableStack(batch_size=B, stack_depth=D, element_dim=E)

    prev_stack = torch.zeros(B, D, E)
    prev_strengths = torch.zeros(B, D)
    v = torch.randn(B, E)
    s = torch.ones(B)

    new_stack, new_strengths = stack.push(v, s, prev_stack, prev_strengths)
    read = stack.read(new_stack, new_strengths)
    assert torch.allclose(read, v, atol=1e-6)

    popped, after_stack, after_strengths = stack.pop(s, new_stack, new_strengths)
    assert torch.allclose(popped, v, atol=1e-6)
    assert torch.allclose(after_stack, torch.zeros_like(after_stack))
    assert torch.allclose(after_strengths, torch.zeros_like(after_strengths))


def test_act_forward_shapes_and_ponder_cost():
    from exercise import AdaptiveComputationTime

    torch.manual_seed(0)
    B, H = 4, 6
    act = AdaptiveComputationTime(hidden_size=H, max_steps=5, ponder_cost=0.01)
    x0 = torch.randn(B, H)
    out, cost = act(x0)
    assert out.shape == (B, H)
    assert cost.shape == (B,)
    assert torch.all(cost >= 0)
    assert torch.all(cost <= 6.0)  # loose upper bound


def test_copy_task_basic_properties():
    from exercise import BenchmarkTasks

    torch.manual_seed(0)
    x, y = BenchmarkTasks.copy_task(batch_size=3, seq_len=5, vocab_size=11)
    assert x.shape == (3, 11)  # 2*seq_len + 1
    assert y.shape == (3, 11)
    # target last seq_len tokens should equal input first seq_len tokens
    assert torch.equal(y[:, -5:], x[:, :5])

