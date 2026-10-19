"""합성 val에서 전처리 기능을 점검한다. CNN 정확도·최종 탐지 지표 평가는 별도다."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from preprocess import Config, PROJECT, load_grayscale, preprocess, save_result


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def select_scenes(dataset, per_count):
    """개수별로 가능한 줄 수를 균등 선택한다. 선택에만 정답 메타데이터 사용."""
    used = Counter()
    records = []
    with (dataset / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            count, row = record["count"], record["row_count"]
            rows = min(4, count) if count else 1
            quota = per_count // rows + (row <= per_count % rows) if count else float("inf")
            key = (count, row)
            if used[key] < quota:
                records.append(record)
                used[key] += 1
    return records


def check(dataset, output, per_count=16, config=None):
    config = config or Config()
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    if metadata["split"] != "tuning":
        raise ValueError("전처리 조정에는 tuning/validation만 사용합니다.")
    if hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest() != metadata["manifest_sha256"]:
        raise ValueError("합성셋 manifest 해시 불일치")
    output.mkdir(parents=True, exist_ok=False)
    records = select_scenes(dataset, per_count)
    groups, examples, failures_saved = defaultdict(Counter), set(), 0
    with (output / "cases.jsonl").open("w", encoding="utf-8") as cases:
        for index, record in enumerate(records, 1):
            path = (dataset / record["image"]).resolve()
            if not path.is_relative_to(dataset.resolve()):
                raise ValueError("이미지 경로가 데이터셋 밖을 가리킵니다.")
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"이미지 해시 불일치: {record['id']}")
            image = load_grayscale(path)
            result = preprocess(image, config)  # 정답 좌표·숫자는 전달하지 않음.
            expected = record["digits"][:config.max_digits]
            count_ok = len(result.candidates) == len(expected)
            # 기능 검사: 읽기 순서대로 비교. 계획의 최종 일대일 채점기를 대신하지 않는다.
            ordered_iou = [iou(c.bbox, d["bbox"]) for c, d in zip(result.candidates, expected)] if count_ok else []
            order_ok = count_ok and all(value >= 0.5 for value in ordered_iou)
            overflow_ok = result.overflow == record["overflow_expected"]
            count = record["count"]
            group = groups[count]
            group["scenes"] += 1
            group["count_matches"] += int(count_ok)
            group["reading_order_box_matches"] += int(order_ok)
            group["overflow_flag_matches"] += int(overflow_ok)
            cases.write(json.dumps({
                "id": record["id"], "count": count, "rows": record["row_count"],
                "detected": len(result.candidates), "excess": len(result.excess),
                "count_matches": count_ok, "reading_order_box_matches": order_ok,
                "ordered_iou": ordered_iou, "overflow_flag_matches": overflow_ok,
                "bbox": [c.bbox for c in result.candidates],
            }) + "\n")
            if count in (1, 4, 8, 16) and count not in examples and record["row_count"] == {1: 1, 4: 2, 8: 3, 16: 4}[count]:
                save_result(image, result, config, output / "examples" / f"n{count:02d}")
                examples.add(count)
            if not (count_ok and order_ok and overflow_ok) and failures_saved < 4:
                save_result(image, result, config, output / "failures" / record["id"])
                failures_saved += 1
            if index % 64 == 0:
                print(f"Checked {index}/{len(records)} scenes", flush=True)
    totals = Counter()
    for group in groups.values():
        totals.update(group)
    report = {
        "scope": "Preprocessing functional probe only; no CNN inference or final accuracy claim",
        "selection": "balanced by digit count and row count; all blank scenes",
        "per_count": per_count, "dataset": str(dataset), "config": asdict(config),
        "manifest_sha256": metadata["manifest_sha256"],
        "box_check": "reading-order pairs with IoU>=0.5; not the final label-blind optimal matcher",
        "totals": dict(totals), "groups": {str(key): dict(value) for key, value in sorted(groups.items())},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "data" / "mnist_scenes" / "tuning")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "build" / "preprocessing" / "validation")
    parser.add_argument("--per-count", type=int, default=16)
    parser.add_argument("--binarization", choices=("adaptive", "otsu"), default="adaptive")
    args = parser.parse_args()
    if args.per_count < 1:
        parser.error("per-count must be positive")
    if args.output_dir.exists():
        parser.error(f"output already exists: {args.output_dir}")
    check(args.dataset, args.output_dir, args.per_count, Config(binarization=args.binarization))
