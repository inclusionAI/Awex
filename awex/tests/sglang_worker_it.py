"""GPU integration check for stock SGLang + AWEX's multiprocessing patch.

Run each topology in a fresh process, for example:
python -m awex.tests.sglang_worker_it --tp 2 --pp 2 --model-dir /path/to/tiny-model
The model is synthetic; no checkpoint downloads are needed.
"""

import argparse
import json
import os


def inspect_worker(**kwargs):
    import torch
    import torch.distributed as dist

    ctx = kwargs["model_context"]
    value = torch.tensor([ctx["global_rank"] + 1], device="cuda", dtype=torch.float32)
    dist.all_reduce(value, group=ctx["scheduler"].world_group.device_group)
    return {
        "pid": os.getpid(),
        "tp": ctx["tp_rank"],
        "pp": ctx["pp_rank"],
        "sum": value.item(),
        "params": len(list(kwargs["model"].named_parameters())),
    }


def change_weight(restore=False, **kwargs):
    import torch

    scheduler = kwargs["model_context"]["scheduler"]
    name, parameter = next(kwargs["model"].named_parameters())
    with torch.no_grad():
        if restore:
            assert torch.all(parameter == 0.25).item()
            parameter.copy_(scheduler._awex_test_saved)
            del scheduler._awex_test_saved
        else:
            scheduler._awex_test_saved = parameter.detach().clone()
            parameter.fill_(0.25)
    return name


def fail_one_rank(**kwargs):
    if kwargs["model_context"]["global_rank"] == 0:
        raise ValueError("intentional callback failure")
    return None


def main():
    import sglang
    from transformers import Qwen3Config

    from awex.config import InferenceConfig
    from awex.engine.sglang import SGLangEngine
    from awex.meta.infer_meta_resolver import InferParamMetaResolver
    from awex.sglang_patch import patch_sglang

    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--dp-attention", action="store_true")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        architectures=["Qwen3ForCausalLM"],
    ).save_pretrained(args.model_dir)
    patch_sglang()
    patch_sglang()  # Idempotent even when several AWEX callers configure it.
    engine = sglang.Engine(
        model_path=args.model_dir,
        load_format="dummy",
        skip_tokenizer_init=True,
        tp_size=args.tp,
        pp_size=args.pp,
        dp_size=args.dp,
        enable_dp_attention=args.dp_attention,
        attention_backend="triton",
        dtype="bfloat16",
        mem_fraction_static=0.1,
        max_total_tokens=256,
        context_length=256,
        max_running_requests=4,
        chunked_prefill_size=128,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        random_seed=42,
        log_level="warning",
    )
    try:
        adapter = SGLangEngine(
            InferenceConfig.from_sgl_engine(engine, comm_backend="file"), engine
        )
        adapter.initialize()
        before = engine.generate(
            input_ids=[1, 2, 3], sampling_params={"temperature": 0, "max_new_tokens": 4}
        )
        ranks = adapter.execute_task_in_model_worker(inspect_worker)
        world_size = args.tp * args.pp
        assert len(ranks) == world_size, ranks
        assert [(r["pp"], r["tp"]) for r in ranks] == [
            (pp, tp) for pp in range(args.pp) for tp in range(args.tp)
        ]
        assert len({r["pid"] for r in ranks}) == world_size
        assert all(r["pid"] != os.getpid() and r["params"] > 0 for r in ranks)
        assert all(r["sum"] == world_size * (world_size + 1) / 2 for r in ranks)
        adapter.execute_task_in_model_worker(change_weight)
        adapter.execute_task_in_model_worker(change_weight, restore=True)
        try:
            adapter.execute_task_in_model_worker(fail_one_rank)
        except RuntimeError as exc:
            assert "intentional callback failure" in str(exc), exc
        else:
            raise AssertionError("worker exception was lost")
        assert (
            adapter.execute_task_in_model_worker(lambda **kw: 17) == [17] * world_size
        )
        if args.dp == 1 or args.dp_attention:
            metadata = InferParamMetaResolver(adapter)
            assert metadata.get_parameters_meta()
        after = engine.generate(
            input_ids=[1, 2, 3], sampling_params={"temperature": 0, "max_new_tokens": 4}
        )
        assert before["output_ids"] == after["output_ids"], (before, after)
        print(
            "AWEX_SGLANG_PASS "
            + json.dumps(
                {
                    "sglang": sglang.__version__,
                    "topology": vars(args),
                    "workers": ranks,
                    "output_ids": after["output_ids"],
                }
            ),
            flush=True,
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
