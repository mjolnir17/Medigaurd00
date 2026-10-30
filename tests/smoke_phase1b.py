"""Temporary manual check for Phase 1b. Replaced by real tests in the testing phase."""
from mediguard.config import IMAGENET_MEAN, IMAGENET_STD
from mediguard.data.dataset import DatasetError, collect_samples, make_loader
from mediguard.disturbances.registry import ALL, build_transform


def first_batch(data, disturbance=None, param=None):
    t = build_transform(224, IMAGENET_MEAN, IMAGENET_STD, disturbance, param)
    x, _ = next(iter(make_loader(data.samples, t, num_workers=0)))
    return x


if __name__ == "__main__":
    data = collect_samples("data/chest_xray")
    print("root   :", data.root)
    print("classes:", data.class_names, data.class_counts)

    clean = first_batch(data)
    print(f"{'clean':<22} shape={tuple(clean.shape)} mean={clean.mean():.3f} std={clean.std():.3f}")
    for key, dist in ALL.items():
        for level in (dist.levels[0], dist.levels[-1]):
            x = first_batch(data, dist, level.param)
            print(f"{key + ' ' + level.label:<22} shape={tuple(x.shape)} mean={x.mean():.3f} std={x.std():.3f}")

    try:
        collect_samples("mediguard/data")
    except DatasetError as e:
        print("bad folder rejected:", e)
