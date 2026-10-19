"""합성셋의 분할·배분·좌표·간격·원본 픽셀·해시를 독립 검사한다."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent


def check(directory):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete", "생성이 완료되지 않은 데이터셋"
    for filename, key in (("manifest.jsonl", "manifest_sha256"), ("source_indices.json", "source_indices_sha256")):
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == metadata[key], filename
    source = json.loads((directory / "source_indices.json").read_text(encoding="utf-8"))
    tuning = metadata["split"] == "tuning"
    assert metadata["split"] in ("tuning", "final")
    if metadata["format_version"] >= 2:
        expected_role = "validation" if tuning else "test"
        assert metadata["task"] == "multi_digit_recognition"
        assert metadata["evaluation_split"] == expected_role
        assert metadata["source_subset"] == source["source_subset"] == f"single_digit_{expected_role}"
    expected_indices = source["indices"] if tuning else list(range(10_000))
    split_recomputed = not tuning
    if tuning:
        # 설치된 torch가 있으면 학습 분할까지 재계산한다. 없으면 생성 시
        # load_mnist(seed=42)에서 저장한 원본 ID와 해시로 픽셀 검사를 수행한다.
        try:
            import torch
            if not hasattr(torch, "randperm"):
                raise ImportError("torch runtime unavailable")
        except (ImportError, OSError):
            pass
        else:
            expected_indices = torch.randperm(60_000, generator=torch.Generator().manual_seed(42)).tolist()[55_000:]
            split_recomputed = True
    assert source["indices"] == expected_indices, "학습 시 사용한 validation/test 분할과 불일치"
    assert len(expected_indices) == (5000 if tuning else 10000)
    assert len(set(expected_indices)) == len(expected_indices)
    assert all(0 <= index < (60000 if tuning else 10000) for index in expected_indices)
    assert source["subset"] == metadata["split"]
    assert source["split_seed"] == (42 if tuning else None)
    assert source["mnist_split"] == ("train" if tuning else "test")
    prefix = "train" if tuning else "t10k"
    raw = PROJECT / "data" / "MNIST" / "raw"
    image_bytes = (raw / f"{prefix}-images-idx3-ubyte").read_bytes()
    label_bytes = (raw / f"{prefix}-labels-idx1-ubyte").read_bytes()
    magic, size, height, width = struct.unpack(">IIII", image_bytes[:16])
    label_magic, label_size = struct.unpack(">II", label_bytes[:8])
    assert (magic, label_magic, height, width) == (2051, 2049, 28, 28)
    assert size == label_size == (60000 if tuning else 10000)
    assert len(image_bytes) == 16 + size * height * width and len(label_bytes) == 8 + size
    images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(size, height, width)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
    allowed = set(expected_indices)
    geometry = metadata["geometry"]
    counts, group_labels, group_rows = Counter(), defaultdict(Counter), defaultdict(Counter)
    group_sources, group_sizes = defaultdict(set), defaultdict(Counter)
    group_positions = defaultdict(Counter)
    seen_ids, seen_paths, cycles = set(), set(), defaultdict(set)
    total_digits = 0
    records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    for record in records:
        assert record["id"] not in seen_ids
        assert record["image"] not in seen_paths
        seen_ids.add(record["id"])
        seen_paths.add(record["image"])
        path = (directory / record["image"]).resolve()
        assert path.is_relative_to(directory.resolve()), "데이터셋 밖을 가리키는 이미지 경로"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], record["id"]
        with Image.open(path) as image:
            assert image.mode == "L"
            pixels = np.array(image)
        assert pixels.shape == (geometry["height"], geometry["width"])
        assert (record["height"], record["width"]) == pixels.shape
        assert len(record["digits"]) == record["count"]
        assert record["overflow_expected"] == (record["count"] > 16)
        if record["kind"] == "blank":
            assert record["count"] == 0 and record["row_count"] == 0
            assert np.all(pixels == 255)
            assert not record["per_row_counts"] and not record["row_baselines"]
        else:
            assert 1 <= record["row_count"] <= min(4, record["count"])
            assert sum(record["per_row_counts"]) == record["count"]
            assert min(record["per_row_counts"]) >= 1
        reconstructed = np.full_like(pixels, 255)
        rows = defaultdict(list)
        covered_ink = 0
        for order, digit in enumerate(record["digits"]):
            index, label = digit["source_index"], digit["label"]
            assert digit["id"] == order
            assert index in allowed and int(labels[index]) == label
            assert digit["source_split"] == source["mnist_split"]
            cycle_key = (label, digit["source_cycle"])
            assert index not in cycles[cycle_key], "비복원 순환 내 원본 중복"
            cycles[cycle_key].add(index)
            assert 0 <= digit["row"] < record["row_count"]
            assert geometry["min_size"] <= digit["size"] <= geometry["max_size"]
            x0, y0, x1, y1 = digit["bbox"]
            assert geometry["margin"] <= x0 < x1 <= geometry["width"] - geometry["margin"]
            assert geometry["margin"] <= y0 < y1 <= geometry["height"] - geometry["margin"]
            ink = pixels[y0:y1, x0:x1] <= 127
            ys, xs = np.nonzero(ink)
            assert len(xs) and (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) == (0, 0, x1 - x0, y1 - y0)
            covered_ink += int(ink.sum())
            px0, py0, px1, py1 = digit["canvas_bbox"]
            assert px1 - px0 == py1 - py0 == digit["size"]
            assert 0 <= px0 < px1 <= geometry["width"] and 0 <= py0 < py1 <= geometry["height"]
            assert py0 - record["row_baselines"][digit["row"]] == digit["jitter_y"]
            assert abs(digit["jitter_y"]) <= geometry["jitter"]
            resized = np.asarray(Image.fromarray(images[index]).resize((digit["size"], digit["size"]), Image.Resampling.BILINEAR))
            reconstructed[py0:py1, px0:px1] = np.minimum(reconstructed[py0:py1, px0:px1], 255 - resized)
            rows[digit["row"]].append(digit)
            group_labels[record["count"]][label] += 1
            group_sources[record["count"]].add(index)
            group_sizes[record["count"]][digit["size"]] += 1
            grid_x = min(3, ((x0 + x1) * 4) // (2 * geometry["width"]))
            grid_y = min(3, ((y0 + y1) * 4) // (2 * geometry["height"]))
            group_positions[record["count"]][f"{grid_x},{grid_y}"] += 1
        assert np.array_equal(reconstructed, pixels), "원본 ID·배치와 PNG 불일치"
        assert covered_ink == int((pixels <= 127).sum()), "정답 상자 밖에 숫자 획 존재"
        order = [(digit["row"], digit["column"]) for digit in record["digits"]]
        assert order == sorted(order), "정답의 읽기 순서 불일치"
        previous_bottom = None
        for row_number in range(record["row_count"]):
            row = rows[row_number]
            assert len(row) == record["per_row_counts"][row_number]
            for column, digit in enumerate(row):
                assert digit["column"] == column
                if column:
                    assert digit["bbox"][0] - row[column - 1]["bbox"][2] >= geometry["digit_gap"]
            top, bottom = min(d["bbox"][1] for d in row), max(d["bbox"][3] for d in row)
            if previous_bottom is not None:
                assert top - previous_bottom >= geometry["row_gap"]
            previous_bottom = bottom
        counts[record["kind"]] += 1
        group_rows[record["count"]][record["row_count"]] += 1
        total_digits += record["count"]
        if len(seen_ids) % 1000 == 0:
            print(f"Checked {len(seen_ids)} scenes", flush=True)
    assert dict(counts) == {key: value for key, value in metadata["scene_counts"].items() if value}
    assert len(records) == metadata["total_scenes"] and total_digits == metadata["total_digits"]
    assert len(set().union(*group_sources.values())) == metadata["unique_source_images_used"]
    per_count = metadata["scene_counts"]["regular"] // 16
    for count in range(1, 17):
        assert sum(group_rows[count].values()) == per_count
    for count in (17, 20):
        assert sum(group_rows[count].values()) == metadata["scene_counts"]["overflow"] // 2
    for count, distribution in group_labels.items():
        values = [distribution[digit] for digit in range(10)]
        assert max(values) - min(values) <= 1, f"숫자 종류 배분 불균형: N={count}"
        frequencies = [group_rows[count][row] for row in range(1, min(4, count) + 1)]
        assert max(frequencies) - min(frequencies) <= 1, f"줄 수 배분 불균형: N={count}"
        group = f"overflow_{count}" if count > 16 else f"n{count:02d}"
        assert {str(key): value for key, value in distribution.items()} == metadata["label_counts"][group]
        assert {str(key): value for key, value in group_rows[count].items()} == metadata["row_counts"][group]
    result = {
        "status": "pass", "scenes": len(records), "digits": total_digits,
        "task": "multi_digit_recognition", "evaluation_split": "validation" if tuning else "test",
        "checks": ["source_index_manifest", "source_labels", "nonreplacement_cycles", "class_and_row_balance", "pixel_reconstruction", "bbox_threshold", "spacing_and_margins", "hashes"],
        "split_recomputed": split_recomputed,
        "split_check": "recomputed" if split_recomputed else "saved training validation indices and hash checked; PyTorch split not independently recomputed",
        "scope": "Dataset integrity only; detection/classification accuracy has not been measured.",
        "coverage": {
            str(count): {
                "scenes": sum(group_rows[count].values()),
                "digits": sum(group_labels[count].values()),
                "unique_source_images": len(group_sources[count]),
                "label_counts": dict(sorted(group_labels[count].items())),
                "row_counts": dict(sorted(group_rows[count].items())),
                "size_counts": dict(sorted(group_sizes[count].items())),
                "position_grid_4x4": dict(sorted(group_positions[count].items())),
            }
            for count in sorted(group_labels)
        },
    }
    (directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "coverage"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=PROJECT / "data" / "mnist_scenes" / "tuning")
    check(parser.parse_args().directory)
