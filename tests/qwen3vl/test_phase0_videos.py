from scripts.render_phase0_videos import episode_frame_paths, select_outcome_episodes


def episode(identifier: str, success: float, paths: list[str]):
    return {
        "episode_id": identifier,
        "final_metrics": {"success": success},
        "steps": [{"history": {"rgb_paths": [path]}} for path in paths],
    }


def test_outcome_selection_is_seeded_and_capped():
    episodes = [episode(str(index), 1.0 if index % 2 else 0.0, [f"/{index}.npy"]) for index in range(8)]

    selected = select_outcome_episodes(episodes, samples_per_outcome=3, seed=7)

    assert len(selected["success"]) == 3
    assert len(selected["failure"]) == 3
    assert selected == select_outcome_episodes(episodes, samples_per_outcome=3, seed=7)


def test_episode_frame_paths_uses_only_latest_distinct_frame():
    item = {
        "steps": [
            {"history": {"rgb_paths": ["/a.npy"]}},
            {"history": {"rgb_paths": ["/a.npy", "/b.npy"]}},
            {"history": {"rgb_paths": ["/a.npy", "/b.npy"]}},
        ]
    }

    assert episode_frame_paths(item) == ["/a.npy", "/b.npy"]
