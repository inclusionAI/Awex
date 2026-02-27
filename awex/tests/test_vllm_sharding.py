from awex.sharding.vllm_sharding import get_vllm_rank_info


def test_get_vllm_rank_info_contains_cp_fields():
    model_context = {
        "tp_size": 2,
        "tp_rank": 1,
        "dp_size": 1,
        "dp_rank": 0,
        "pp_rank": 0,
        "pp_size": 1,
        "ep_rank": 0,
        "ep_size": 1,
        "ep_tp_rank": 0,
        "ep_tp_size": 1,
        "attn_tp_rank": 1,
        "attn_tp_size": 2,
        "attn_dp_rank": 0,
        "world_size": 4,
        "global_rank": 1,
        "local_rank": 1,
        "cp_rank": 1,
        "cp_size": 2,
        "cp_mode": "ulysses",
    }

    rank_info = get_vllm_rank_info(model_context, engine_rank=0)
    assert rank_info.cp_rank == 1
    assert rank_info.cp_size == 2
    assert rank_info.cp_mode == "ulysses"
