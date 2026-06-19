from evaluation.score import bootstrap_mean_ci, score_results


def test_scores_paired_experiment_results() -> None:
    rows = [
        {
            "task_id": "1",
            "arm": "raw",
            "gold_files": ["a.py"],
            "predicted_files": ["b.py", "a.py"],
            "patch_success": False,
            "iterations": 4,
            "runtime_seconds": 10,
        },
        {
            "task_id": "1",
            "arm": "evidence",
            "gold_files": ["a.py"],
            "predicted_files": ["a.py"],
            "patch_success": True,
            "iterations": 2,
            "runtime_seconds": 9,
        },
    ]
    report = score_results(rows)
    assert report["arms"]["raw"]["top5_localization"] == 1.0
    assert report["arms"]["evidence"]["patch_success"] == 1.0
    assert report["paired"]["patch_success_delta"] == 1.0
    assert report["paired"]["median_iteration_delta"] == -2


def test_bootstrap_interval_is_deterministic() -> None:
    assert bootstrap_mean_ci([1.0, 1.0, 1.0], samples=100) == (1.0, 1.0)
