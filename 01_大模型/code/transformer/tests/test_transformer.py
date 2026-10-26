"""验证注意力公式、信息隔离、反向传播和真正的自回归生成。"""

import pytest
import torch
from torch import nn

from model import (
    BOS,
    EOS,
    PAD,
    Config,
    SinusoidalPosition,
    Transformer,
    greedy_decode,
    scaled_dot_product_attention,
    source_mask,
    target_mask,
)
from train_copy import make_batch


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return Transformer(Config(d_model=16, n_heads=2, d_ff=32, n_layers=1)).eval()


def test_attention_matches_pytorch_reference():
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(2, 2, 3, 4, generator=generator)
    k = torch.randn(2, 2, 5, 4, generator=generator)
    v = torch.randn(2, 2, 5, 4, generator=generator)
    mask = torch.tensor([True, True, False, True, False])[None, None, None, :]
    context, weights = scaled_dot_product_attention(q, k, v, mask)
    reference = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(context, reference)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2, 3))
    assert torch.count_nonzero(weights[..., [2, 4]]) == 0


def test_uniform_scores_average_only_visible_values():
    q = k = torch.zeros(1, 1, 3, 2)
    v = torch.tensor([[[[2.0, 4.0], [6.0, 8.0], [100.0, 200.0]]]])
    mask = torch.tensor([True, True, False])[None, None, None, :]
    context, _ = scaled_dot_product_attention(q, k, v, mask)
    torch.testing.assert_close(context, torch.tensor([[[[4.0, 6.0]]]]).expand_as(v))


def test_fully_masked_attention_is_finite_and_zero():
    q = torch.ones(1, 1, 2, 4, requires_grad=True)
    context, weights = scaled_dot_product_attention(
        q, q, q, torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    )
    assert torch.isfinite(context).all()
    assert torch.count_nonzero(context) == 0
    assert torch.count_nonzero(weights) == 0
    context.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_target_mask_combines_causal_and_padding():
    mask = target_mask(torch.tensor([[BOS, 3, PAD]]))[0, 0]
    expected = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_future_target_tokens_cannot_change_past_logits(model):
    src = torch.tensor([[3, 4, EOS, PAD]])
    tgt = torch.tensor([[BOS, 3, 4, 5]])
    changed = torch.tensor([[BOS, 3, 9, 10]])
    with torch.no_grad():
        original_logits = model(src, tgt)
        changed_logits = model(src, changed)
    torch.testing.assert_close(original_logits[:, :2], changed_logits[:, :2])


def test_source_padding_memory_cannot_change_decoder_output(model):
    src = torch.tensor([[3, EOS, PAD, PAD]])
    tgt = torch.tensor([[BOS, 3]])
    mask = source_mask(src)
    with torch.no_grad():
        memory = model.encode(src, mask)
        noisy_memory = memory.clone()
        noisy_memory[src == PAD] = 10000
        original = model.decode(tgt, memory, mask, target_mask(tgt))
        changed = model.decode(tgt, noisy_memory, mask, target_mask(tgt))
    torch.testing.assert_close(original, changed)


def test_appending_source_padding_does_not_change_output(model):
    src = torch.tensor([[3, 4, EOS]])
    padded = torch.tensor([[3, 4, EOS, PAD, PAD]])
    tgt = torch.tensor([[BOS, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(model(src, tgt), model(padded, tgt))


def test_cross_attention_supports_different_source_target_lengths(model):
    logits = model(torch.tensor([[3, 4, 5, EOS]]), torch.tensor([[BOS, 3]]))
    assert logits.shape == (1, 2, model.config.vocab_size)


def test_batch_shift_and_eos():
    src, tgt, labels = make_batch(8, 5, 16, torch.Generator().manual_seed(42))
    assert torch.equal(tgt[:, 1:], labels[:, :-1])
    assert (tgt[:, 0] == BOS).all()
    assert torch.equal(src, labels)
    assert ((labels == EOS).sum(-1) == 1).all()
    assert (src == PAD).any()


def test_gradients_reach_both_embeddings_and_ignore_pad(model):
    src, tgt, labels = make_batch(8, 5, 16, torch.Generator().manual_seed(42))
    logits = model(src, tgt)
    nn.functional.cross_entropy(
        logits.reshape(-1, 16), labels.reshape(-1), ignore_index=PAD
    ).backward()
    for embedding in (model.src_embedding, model.tgt_embedding):
        assert embedding.table.weight.grad[3:].abs().sum() > 0
        assert torch.count_nonzero(embedding.table.weight.grad[PAD]) == 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_greedy_decode_stops_at_eos_and_restores_mode(model):
    # 控制输出头，只测生成循环的 EOS 终止；不是模型学会任务的证据。
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[EOS] = 10
    model.train()
    result = greedy_decode(model, torch.tensor([[3, EOS], [4, EOS]]), 6)
    assert result.tolist() == [[BOS, EOS], [BOS, EOS]]
    assert model.training


def test_greedy_decode_respects_generation_limit(model):
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[3] = 10
    result = greedy_decode(model, torch.tensor([[3, EOS]]), 3)
    assert result.tolist() == [[BOS, 3, 3, 3]]


def test_optimizer_can_overfit_a_tiny_copy_batch():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = Transformer(Config(d_model=32, n_heads=2, d_ff=64, n_layers=1))
    batch = make_batch(4, 3, 16, torch.Generator().manual_seed(43))
    src, tgt, labels = batch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss():
        logits = model(src, tgt)
        return nn.functional.cross_entropy(
            logits.reshape(-1, 16), labels.reshape(-1), ignore_index=PAD
        )

    initial = loss().item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        current = loss()
        current.backward()
        optimizer.step()
    assert loss().item() < initial * 0.1
    generated = greedy_decode(model, src, labels.size(1))
    for prediction, expected in zip(generated[:, 1:].tolist(), labels.tolist()):
        expected = [token for token in expected if token != PAD]
        assert prediction[: len(expected)] == expected


def test_positions_are_buffers_and_enforce_length_limit():
    position = SinusoidalPosition(d_model=7, max_len=4, dropout=0)
    assert list(position.parameters()) == []
    assert position(torch.zeros(1, 4, 7)).shape == (1, 4, 7)
    with pytest.raises(ValueError, match="max_len"):
        position(torch.zeros(1, 5, 7))


def test_invalid_head_dimension_is_rejected():
    with pytest.raises(ValueError, match="整除"):
        Config(d_model=7, n_heads=2)
