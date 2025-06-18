from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.notebook import tqdm

from hepattn.experiments.cld.data import CLDDataset
from hepattn.experiments.cld.plot_event import plot_cld_event_pre_vs_post, plot_cld_event_reconstruction

plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True


def main():
    config_path = Path("/home/syw24/ftag/hepattn/logs/CLD_TRKECALHCAL_16_96_TF_charged_10MeV_F16_manypass_20250529-T024522/config.yaml")
    config = yaml.safe_load(config_path.read_text())["data"]
    config["dirpath"] = Path("/share/rcifdata/maxhart/data/cld/prepped/train/")
    config_del_keys = [
        "train_dir",
        "test_dir",
        "val_dir",
        "num_train",
        "num_test",
        "num_val",
        "num_workers",
        "batch_size",
        "pin_memory",
    ]
    for key in config_del_keys:
        config.pop(key)

    config_no_cuts = config.copy()
    config_no_cuts.update({
        "charged_particle_min_num_hits": {},
        "charged_particle_max_num_hits": {},
        "neutral_particle_min_num_hits": {},
        "neutral_particle_max_num_hits": {},
        "particle_hit_min_p_ratio": {},
        "particle_hit_deflection_cuts": {},
        "particle_hit_separation_cuts": {},
        "truth_filter_hits": [],
    })

    dataset = CLDDataset(**config)
    original_dataset = CLDDataset(**config_no_cuts)

    truth_outdir = Path("/share/gpu1/syw24/plots/cld/event_cleaning/truth")
    trkr_outdir = Path("/share/gpu1/syw24/plots/cld/event_cleaning/compare_trkr")
    vtxd_outdir = Path("/share/gpu1/syw24/plots/cld/event_cleaning/compare_vtxd")

    num_events = 100
    rng = np.random.default_rng(seed=42)
    rand_events = rng.choice(len(dataset), size=num_events, replace=False)

    hits = ["trkr", "ecal", "hcal"]
    # Spec for event displays
    axes_spec = [
        {"x": "pos.x", "y": "pos.y", "px": "mom.x", "py": "mom.y", "input_names": hits},
        {"x": "pos.z", "y": "pos.y", "px": "mom.z", "py": "mom.y", "input_names": hits},
    ]

    vtxd_hits = ["vtxd"]
    vtxd_axes_spec = [
        {"x": "pos.x", "y": "pos.y", "px": "mom.x", "py": "mom.y", "input_names": vtxd_hits},
        {"x": "pos.z", "y": "pos.y", "px": "mom.z", "py": "mom.y", "input_names": vtxd_hits},
    ]

    for idx in tqdm(range(num_events)):
        evt_idx = rand_events[idx]

        sample_id = dataset.sample_ids[evt_idx]
        inputs, targets = dataset[evt_idx]

        original_sample_id = original_dataset.sample_ids[evt_idx]
        original_inputs, original_targets = original_dataset[evt_idx]

        assert original_sample_id == sample_id, "id mismatch"

        fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
        fig.savefig(truth_outdir / Path(f"post_truth_id_{sample_id}.png"))
        plt.close(fig)

        fig = plot_cld_event_pre_vs_post(original_inputs, inputs, original_targets, targets, axes_spec)
        fig.savefig(trkr_outdir / Path(f"pre_vs_post_trkr_id_{sample_id}.png"))
        plt.close(fig)

        fig = plot_cld_event_pre_vs_post(original_inputs, inputs, original_targets, targets, vtxd_axes_spec)
        fig.savefig(vtxd_outdir / Path(f"pre_vs_post_vtxd_id_{sample_id}.png"))
        plt.close(fig)

    print(f"Produced {num_events} event display")


if __name__ == "__main__":
    main()
