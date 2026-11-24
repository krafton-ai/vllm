# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.models.raon1_mini import kda_split_decode as split


def qsl(lengths):
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return out


def plan(lengths, prefilling, spec=0):
    return split.plan_split(qsl(lengths), prefilling, sum(lengths), spec)


def test_decode_first_batch_is_split():
    p = plan([1, 1, 1, 1024, 7], [False, False, False, True, True])
    assert p == split.SplitPlan(num_decodes=3, num_prefills=2, num_prefill_tokens=1031)


def test_one_token_prefill_chunk_is_not_a_decode():
    # Query length 1 but still prefilling: the state may not be initialized.
    p = plan([1, 1, 1, 500], [False, False, True, True])
    assert p == split.SplitPlan(num_decodes=2, num_prefills=2, num_prefill_tokens=501)
    assert plan([1, 1, 500], [True, False, True]) is None


def test_multi_token_row_goes_to_prefill_side():
    p = plan([1, 2, 64], [False, False, True])
    assert p == split.SplitPlan(num_decodes=1, num_prefills=2, num_prefill_tokens=66)


@pytest.mark.parametrize(
    ("lengths", "prefilling", "spec"),
    [
        ([1, 1, 1], [False] * 3, 0),  # decode only
        ([8, 9], [True, True], 0),  # prefill only
        ([1, 1, 64], [False, False, True], 1),  # speculative rows present
        ([1, 64, 1], [False, True, False], 0),  # decode after prefill
        ([64, 1], [True, False], 0),  # decode as suffix, not first
        ([1, 0, 64], [False, True, True], 0),  # zero-length padding row
    ],
)
def test_ineligible_layouts(lengths, prefilling, spec):
    assert plan(lengths, prefilling, spec) is None


def test_missing_or_inconsistent_metadata():
    assert split.plan_split(qsl([1, 8]), None, 9, 0) is None
    assert split.plan_split(qsl([1, 8]), [False, True], 10, 0) is None
    assert split.plan_split(qsl([1, 8]), [False], 9, 0) is None
    assert split.plan_split([1, 2, 10], [False, True], 10, 0) is None


def test_enabled_env(monkeypatch):
    monkeypatch.delenv("VLLM_RAON_KDA_SPLIT_DECODE_SUFFIX", raising=False)
    assert split.explicit() is False
    monkeypatch.setenv("VLLM_RAON_KDA_SPLIT_DECODE_SUFFIX", "1")
    assert split.explicit() is True
    monkeypatch.setenv("VLLM_RAON_KDA_SPLIT_DECODE_SUFFIX", "yes")
    with pytest.raises(ValueError):
        split.explicit()


def make_metadata(torch, lengths):
    from vllm.models.raon1_mini.rda_metadata import RaonRDAMetadata

    n = len(lengths)
    q = torch.tensor(qsl(lengths), dtype=torch.int32)
    return RaonRDAMetadata(
        num_prefills=1,
        num_prefill_tokens=sum(lengths) - 1,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=sum(lengths),
        has_initial_state=torch.tensor([True] + [False] * (n - 1)),
        non_spec_query_start_loc=q,
        non_spec_state_indices_tensor=torch.arange(10, 10 + n, dtype=torch.int32),
    ), q


def test_attach_plan_builds_prefill_only_metadata(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("VLLM_RAON_KDA_SPLIT_DECODE_SUFFIX", "1")
    lengths = [1, 1, 40, 17]
    metadata, q = make_metadata(torch, lengths)
    common = SimpleNamespace(
        num_reqs=len(lengths),
        query_start_loc_cpu=q,
        is_prefilling=torch.tensor([False, False, True, True]),
    )
    from vllm.models.raon1_mini import kda_vsplit

    monkeypatch.setattr(kda_vsplit, "enabled", lambda: False)
    split.attach_plan(SimpleNamespace(), common, metadata)

    got = split.get_plan(metadata)
    assert got is not None
    plan_, pm = got
    assert plan_.num_decodes == 2
    assert pm.num_decodes == 0 and pm.num_prefills == 2
    assert pm.num_actual_tokens == 57
    assert pm.non_spec_query_start_loc.tolist() == [0, 40, 57]
    assert pm.non_spec_state_indices_tensor.tolist() == [12, 13]
    assert pm.has_initial_state.tolist() == [False, False]
    # The original step metadata is untouched.
    assert metadata.num_decodes == 1
    assert metadata.non_spec_query_start_loc.tolist() == [0, 1, 2, 42, 59]


def test_attach_plan_off_attaches_nothing(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("VLLM_RAON_KDA_SPLIT_DECODE_SUFFIX", raising=False)
    metadata, q = make_metadata(torch, [1, 8])
    split.attach_plan(SimpleNamespace(), SimpleNamespace(), metadata)
    assert split.get_plan(metadata) is None
