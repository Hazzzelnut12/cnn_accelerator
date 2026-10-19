"""MNIST 원본으로 다중 숫자 판독 평가용 장면과 정답을 생성한다."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

PROJECT = Path(__file__).resolve().parent.parent
SCENES_PER_COUNT = 1000


@dataclass(frozen=True)
class Geometry:
    width: int = 640
    height: int = 480
    min_size: int = 24
    max_size: int = 28
    margin: int = 16
    digit_gap: int = 8
    row_gap: int = 16
    jitter: int = 2
    max_rows: int = 4


class SourcePool:
    """숫자 종류별로 원본을 비복원 추출하고, 소진하면 다시 섞어 순환한다."""

    def __init__(self, images, labels, indices, source_split, seed):
        self.images = images
        self.source_split = source_split
        self.rng = random.Random(seed)
        self.indices = {
            digit: [i for i in indices if int(labels[i]) == digit]
            for digit in range(10)
        }
        if any(not items for items in self.indices.values()):
            raise ValueError("0~9 각 종류의 원본이 필요합니다.")
        self.pending = {digit: [] for digit in range(10)}
        self.cycles = {digit: -1 for digit in range(10)}

    def draw(self, digit):
        if not self.pending[digit]:
            self.pending[digit] = self.indices[digit].copy()
            self.rng.shuffle(self.pending[digit])
            self.cycles[digit] += 1
        index = self.pending[digit].pop()
        return {
            "label": digit,
            "source_split": self.source_split,
            "source_index": int(index),
            "source_cycle": self.cycles[digit],
            "pixels": self.images[index],
        }


def allocation(total, parts, rng, minimum=0):
    """합이 total인 정수 parts개. 각 항의 최솟값은 minimum."""
    remainder = total - parts * minimum
    if parts < 1 or remainder < 0:
        raise ValueError("분할할 수 없는 개수입니다.")
    bars = sorted(rng.sample(range(remainder + parts - 1), parts - 1))
    edges = [-1, *bars, remainder + parts - 1]
    return [edges[i + 1] - edges[i] - 1 + minimum for i in range(parts)]


def ink_box(pixels):
    ys, xs = np.nonzero(pixels >= 128)
    if not len(xs):
        raise ValueError("화소값 128 이상인 숫자 획이 없습니다.")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def make_scene(samples, row_count, geometry, seed):
    """크기 변경만 적용한다. bbox는 밝은 획 >=128, xyxy 반개구간이다."""
    g = geometry
    rng = random.Random(seed)
    for _ in range(128):
        per_row = allocation(len(samples), row_count, rng, minimum=1)
        rows, offset = [], 0
        for amount in per_row:
            row = []
            for sample in samples[offset:offset + amount]:
                side = rng.randint(g.min_size, g.max_size)
                pixels = np.asarray(
                    Image.fromarray(sample["pixels"]).resize(
                        (side, side), Image.Resampling.BILINEAR,
                    ), dtype=np.uint8,
                )
                row.append((sample, pixels, ink_box(pixels), side))
            rows.append(row)
            offset += amount

        row_heights = [max(item[3] for item in row) + 2 * g.jitter for row in rows]
        vertical_slack = (
            g.height - 2 * g.margin - sum(row_heights) - g.row_gap * (row_count - 1)
        )
        widths = [sum(item[2][2] - item[2][0] for item in row) for row in rows]
        horizontal_slack = [
            g.width - 2 * g.margin - width - g.digit_gap * (len(row) - 1)
            for row, width in zip(rows, widths)
        ]
        if vertical_slack < 0 or min(horizontal_slack) < 0:
            continue

        y_spaces = allocation(vertical_slack, row_count + 1, rng)
        row_top = g.margin + y_spaces[0]
        placements, baselines = [], []
        valid = True
        for row_index, row in enumerate(rows):
            baseline = row_top + g.jitter
            baselines.append(baseline)
            x_spaces = allocation(horizontal_slack[row_index], len(row) + 1, rng)
            ink_left = g.margin + x_spaces[0]
            for column, (sample, pixels, local_box, side) in enumerate(row):
                jitter_y = rng.randint(-g.jitter, g.jitter)
                patch_x, patch_y = ink_left - local_box[0], baseline + jitter_y
                if not (0 <= patch_x <= g.width - side and 0 <= patch_y <= g.height - side):
                    valid = False
                    break
                bbox = [
                    patch_x + local_box[0], patch_y + local_box[1],
                    patch_x + local_box[2], patch_y + local_box[3],
                ]
                annotation = {key: value for key, value in sample.items() if key != "pixels"}
                annotation.update(
                    id=len(placements), row=row_index, column=column, bbox=bbox,
                    canvas_bbox=[patch_x, patch_y, patch_x + side, patch_y + side],
                    size=side, jitter_y=jitter_y,
                )
                placements.append((annotation, pixels))
                ink_left = bbox[2] + g.digit_gap + x_spaces[column + 1]
            if not valid:
                break
            row_top += row_heights[row_index] + g.row_gap + y_spaces[row_index + 1]
        if not valid:
            continue

        frame = np.full((g.height, g.width), 255, dtype=np.uint8)
        for item, pixels in placements:
            x0, y0, x1, y1 = item["canvas_bbox"]
            # 흰 배경·검은 숫자로 반전. 옅은 가장자리가 겹치면 진한 쪽을 보존.
            frame[y0:y1, x0:x1] = np.minimum(frame[y0:y1, x0:x1], 255 - pixels)
        return frame, [item for item, _ in placements], per_row, baselines
    raise ValueError(
        f"숫자 {len(samples)}개·{row_count}줄을 {g.width}x{g.height}에 배치할 수 없습니다. "
        "해상도·숫자 크기·간격을 확인하세요."
    )


def load_sources(split, source_manifest=None):
    if source_manifest is not None:
        if split != "tuning":
            raise ValueError("저장된 validation 분할은 조정용에만 사용할 수 있습니다.")
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        prior = json.loads((source_manifest.parent / "metadata.json").read_text(encoding="utf-8"))
        source_hash = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        if prior["source_indices_sha256"] != source_hash or prior["status"] != "complete":
            raise ValueError("저장된 원본 목록의 해시·완료 상태가 올바르지 않습니다.")
        indices = source["indices"]
        if not (
            source["mnist_split"] == "train" and source["subset"] == "tuning"
            and source["split_seed"] == 42 and prior["split"] == "tuning"
            and len(indices) == len(set(indices)) == 5000
            and all(type(index) is int and 0 <= index < 60000 for index in indices)
        ):
            raise ValueError("학습 당시 validation 5,000장의 분할 목록이 필요합니다.")
        raw = PROJECT / "data" / "MNIST" / "raw"
        image_bytes = (raw / "train-images-idx3-ubyte").read_bytes()
        label_bytes = (raw / "train-labels-idx1-ubyte").read_bytes()
        if (
            struct.unpack(">IIII", image_bytes[:16]) != (2051, 60000, 28, 28)
            or struct.unpack(">II", label_bytes[:8]) != (2049, 60000)
            or len(image_bytes) != 16 + 60000 * 28 * 28
            or len(label_bytes) != 8 + 60000
        ):
            raise ValueError("MNIST 원본 IDX 형식·크기가 올바르지 않습니다.")
        images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(60000, 28, 28)
        labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
        return images, labels, indices, "train", None, {
            "method": "saved_training_validation_indices",
            "split_seed": 42, "parent_source_indices_sha256": source_hash,
            "original_split_torch_version": prior["versions"]["torch"],
        }

    import torch
    from torchvision.datasets import MNIST
    from train import load_mnist

    if split == "tuning":
        _, subset = load_mnist(seed=42, augment=False)
        dataset, indices, source_split = subset.dataset, list(subset.indices), "train"
    else:
        dataset = MNIST(root=PROJECT / "data", train=False, download=True)
        indices, source_split = list(range(len(dataset))), "test"
    return (
        dataset.data.numpy(), dataset.targets.numpy(), indices, source_split,
        str(torch.__version__),
        {"method": "training_load_mnist" if split == "tuning" else "official_test"},
    )


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_preview(output, records):
    sheet = Image.new("RGB", (640, 520), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, record in enumerate(records):
        image = Image.open(output / record["image"]).convert("RGB")
        overlay = ImageDraw.Draw(image)
        for digit in record["digits"]:
            x0, y0, x1, y1 = digit["bbox"]
            overlay.rectangle((x0, y0, x1 - 1, y1 - 1), outline="red")
            overlay.text((x0, max(0, y0 - 12)), str(digit["label"]), fill="blue")
        image.thumbnail((320, 240))
        x, y = (slot % 2) * 320, (slot // 2) * 260
        sheet.paste(image, (x, y + 20))
        draw.text((x + 4, y + 3), f"N={record['count']}, rows={record['row_count']}", fill="black")
    sheet.save(output / "preview.png")


def generate(args):
    geometry = Geometry(width=args.width, height=args.height)
    seed = args.seed if args.seed is not None else (42 if args.split == "tuning" else 43)
    per_count = args.scenes_per_count or SCENES_PER_COUNT
    blanks = args.blank_scenes if args.blank_scenes is not None else (20 if args.split == "tuning" else 100)
    overflows = args.overflow_scenes if args.overflow_scenes is not None else (100 if args.split == "tuning" else 10)
    output = args.output_dir or PROJECT / "data" / "mnist_scenes" / args.split
    if output.exists():
        raise FileExistsError(f"기존 평가셋은 덮어쓰지 않습니다: {output}")
    images, labels, indices, source_split, torch_version, source_selection = load_sources(args.split, args.source_manifest)
    output.mkdir(parents=True, exist_ok=False)
    evaluation_split = "validation" if args.split == "tuning" else "test"
    source_info = {
        "mnist_split": source_split, "subset": args.split,
        "source_subset": f"single_digit_{evaluation_split}",
        "source_selection": source_selection,
        "split_seed": 42 if args.split == "tuning" else None, "indices": indices,
    }
    write_json(output / "source_indices.json", source_info)
    source_hash = hashlib.sha256((output / "source_indices.json").read_bytes()).hexdigest()
    rng = random.Random(seed)
    pool = SourcePool(images, labels, indices, source_split, seed ^ 0x4D4E4953)
    previews, label_counts, row_counts, unique_ids = [], {}, {}, set()
    scene_total = digit_total = 0
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for count in [*range(1, 17), 0, 17, 20]:
            amount = blanks if count == 0 else overflows if count > 16 else per_count
            kind = "blank" if count == 0 else "overflow" if count > 16 else "regular"
            group = "blank" if count == 0 else f"overflow_{count}" if count > 16 else f"n{count:02d}"
            image_dir = output / "images" / group
            image_dir.mkdir(parents=True)
            balanced_labels = [i % 10 for i in range(amount * count)]
            rng.shuffle(balanced_labels)
            row_schedule = [1 + i % min(geometry.max_rows, count) for i in range(amount)] if count else [0] * amount
            rng.shuffle(row_schedule)
            group_labels, group_rows = Counter(), Counter()
            for number, rows in enumerate(row_schedule):
                scene_seed = rng.getrandbits(64)
                chosen = balanced_labels[number * count:(number + 1) * count]
                if count:
                    samples = [pool.draw(digit) for digit in chosen]
                    frame, digits, per_row, baselines = make_scene(samples, rows, geometry, scene_seed)
                else:
                    frame = np.full((geometry.height, geometry.width), 255, dtype=np.uint8)
                    digits, per_row, baselines = [], [], []
                relative_path = f"images/{group}/scene_{number:04d}.png"
                Image.fromarray(frame).save(output / relative_path, compress_level=6)
                record = {
                    "id": f"{group}-{number:04d}", "kind": kind, "count": count,
                    "image": relative_path, "sha256": hashlib.sha256((output / relative_path).read_bytes()).hexdigest(),
                    "width": geometry.width, "height": geometry.height, "seed": scene_seed,
                    "row_count": rows, "per_row_counts": per_row, "row_baselines": baselines,
                    "overflow_expected": count > 16, "digits": digits,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                if count in (1, 4, 8, 16) and number == 0:
                    previews.append(record)
                group_labels.update(chosen)
                group_rows[rows] += 1
                unique_ids.update(item["source_index"] for item in digits)
                scene_total += 1
                digit_total += count
            label_counts[group] = dict(sorted(group_labels.items()))
            row_counts[group] = dict(sorted(group_rows.items()))
            print(f"{group}: {amount} scenes", flush=True)

    save_preview(output, previews)
    metadata = {
        "format_version": 2, "status": "complete", "purpose": "evaluation_only_no_retraining",
        "task": "multi_digit_recognition", "evaluation_split": evaluation_split,
        "source_subset": f"single_digit_{evaluation_split}",
        "split": args.split, "seed": seed, "geometry": asdict(geometry),
        "resolution_status": "provisional_640x480" if (args.width, args.height) == (640, 480) else "user_selected",
        "scene_counts": {"regular": per_count * 16, "blank": blanks, "overflow": overflows * 2},
        "total_scenes": scene_total, "total_digits": digit_total, "unique_source_images_used": len(unique_ids),
        "source_split": source_split, "source_subset_size": len(indices),
        "source_indices_sha256": source_hash,
        "manifest_sha256": hashlib.sha256((output / "manifest.jsonl").read_bytes()).hexdigest(),
        "source_reuse": "per-label shuffled nonreplacement cycles; repeated originals are not independent samples",
        "bbox": "xyxy, right/bottom exclusive; bounding ink >=128 after resizing",
        "pixels": "grayscale uint8; white background, black digits; no added rotation/noise/blur",
        "resize": "Pillow BILINEAR; full 28x28 MNIST canvas resized to 24..28 square",
        "label_counts": label_counts, "row_counts": row_counts,
        "versions": {"python": sys.version.split()[0], "torch": torch_version, "numpy": np.__version__, "pillow": pillow_version},
    }
    write_json(output / "metadata.json", metadata)
    print(f"Complete: {output} ({scene_total} scenes, {digit_total} digits)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("tuning", "final"), default="tuning")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-manifest", type=Path, help="저장된 validation 분할로 조정용 재생성 (MNIST 원본 IDX 필요)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scenes-per-count", type=int, help="1~16개 그룹별 장면 수 (기본 1,000)")
    parser.add_argument("--blank-scenes", type=int)
    parser.add_argument("--overflow-scenes", type=int, help="17개·20개별 장면 수 (기본: 조정 100, 최종 10)")
    args = parser.parse_args()
    if min(args.width, args.height) < 64:
        parser.error("width/height must be at least 64")
    if args.scenes_per_count is not None and args.scenes_per_count <= 0:
        parser.error("scenes-per-count must be positive")
    if (args.blank_scenes is not None and args.blank_scenes < 0) or (args.overflow_scenes is not None and args.overflow_scenes < 0):
        parser.error("blank-scenes/overflow-scenes must be nonnegative")
    return args


if __name__ == "__main__":
    generate(parse_args())
