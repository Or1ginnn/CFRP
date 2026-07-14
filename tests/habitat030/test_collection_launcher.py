from scripts.launch_stage1_warmup_collection import build_shards


def test_full_r2r_shards_cover_every_episode_without_overlap():
    shards = build_shards(10819, 100)

    assert len(shards) == 109
    assert shards[0].episode_offset == 0
    assert shards[-1].episode_offset == 10800
    assert shards[-1].episode_count == 19
    covered = [
        episode
        for shard in shards
        for episode in range(
            shard.episode_offset, shard.episode_offset + shard.episode_count
        )
    ]
    assert covered == list(range(10819))
