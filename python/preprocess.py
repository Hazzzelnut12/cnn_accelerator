"""다중 숫자 위치 탐지·읽기 순서·회색조 NCHW 입력 생성. CNN 추론은 수행하지 않는다.

입력: 2차원 uint8(0..255) 또는 float(0..1). 기본은 흰 배경·검은 글씨.
bbox: [x0, y0, x1, y1], 오른쪽·아래 끝 제외. 기본 설정은 Claude v3 참조.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageOps

PROJECT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    binarization: str = "adaptive"
    sensitivity: float = 0.2
    min_height: int = 12
    merge: bool = True
    max_digits: int = 16
    background: str = "white"
    crop_padding: int = 2
    crop_mode: str = "grayscale"

    def __post_init__(self):
        if self.binarization not in ("adaptive", "otsu"):
            raise ValueError("binarization must be adaptive or otsu")
        if self.background not in ("white", "black"):
            raise ValueError("background must be white or black")
        if self.crop_mode not in ("grayscale", "mask"):
            raise ValueError("crop_mode must be grayscale or mask")
        if not 0 < self.sensitivity < 1:
            raise ValueError("sensitivity must be between 0 and 1")
        if not 1 <= self.max_digits <= 16:
            raise ValueError("max_digits must be between 1 and 16")
        if self.min_height < 1 or self.crop_padding < 0:
            raise ValueError("min_height must be positive; padding must be nonnegative")


@dataclass(frozen=True)
class Component:
    bbox: tuple[int, int, int, int]
    area: int
    members: tuple[int, ...]


@dataclass(frozen=True)
class Candidate:
    bbox: tuple[int, int, int, int]  # 팽창·crop 여백 전 검출 획의 최소 상자
    component_bbox: tuple[int, int, int, int]  # 병합용 팽창 마스크의 상자
    crop_bbox: tuple[int, int, int, int]
    area: int  # 팽창 전 검출 획 면적
    members: tuple[int, ...]
    row: int = 0
    column: int = 0
    space_before: bool = False


@dataclass
class Result:
    inputs: np.ndarray  # float32, [N, 1, 28, 28], 검은 배경·밝은 숫자, 0..1
    candidates: list[Candidate]
    excess: list[Candidate]
    dropped: list[dict]
    mask: np.ndarray
    component_count: int
    elapsed_ms: float

    @property
    def overflow(self):
        return bool(self.excess)


def load_grayscale(path):
    """EXIF 방향 적용. 투명 부분은 흰 배경에 합성하고 8bit 회색조로 읽는다."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        white = Image.new("RGBA", image.size, "white")
        return np.array(Image.alpha_composite(white, image).convert("L"))


def grayscale(image):
    gray = np.asarray(image)
    if gray.ndim != 2 or not gray.size:
        raise ValueError("input must be a nonempty 2D grayscale image")
    if gray.dtype == np.uint8:
        return gray.astype(np.float64) / 255
    if not np.issubdtype(gray.dtype, np.floating):
        raise ValueError("input must be uint8 or floating point in 0..1")
    if not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1:
        raise ValueError("floating point input must be finite and in 0..1")
    return gray.astype(np.float64)


def binarize(gray, mode="adaptive", sensitivity=0.2):
    """위치 탐지용 마스크. CNN에 넣을 원본 회색조와 별도로 유지한다."""
    height, width = gray.shape
    if mode == "otsu":
        levels = np.floor(gray * 255 + 0.5).astype(np.uint8)
        hist = np.bincount(levels.ravel(), minlength=256).astype(np.float64)
        if np.count_nonzero(hist) <= 1:
            return np.zeros(gray.shape, dtype=bool)
        weight = np.cumsum(hist)
        total = gray.size
        moment = np.cumsum(hist * np.arange(256))
        valid = (weight > 0) & (weight < total)
        between = np.full(256, -1.0)
        between[valid] = (moment[-1] * weight[valid] - moment[valid] * total) ** 2 / (
            weight[valid] * (total - weight[valid])
        )
        threshold = int(np.argmax(between))
        # Otsu의 아래쪽 군집은 threshold를 포함한다. 순수 흑백의 threshold=0도 포함.
        return levels <= threshold
    if mode != "adaptive":
        raise ValueError("unknown binarization mode")
    radius = max(7, int(np.floor(min(width, height) / 16 + 0.5)))
    summed = np.pad(gray.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    y0 = np.maximum(0, np.arange(height) - radius)
    y1 = np.minimum(height, np.arange(height) + radius + 1)
    x0 = np.maximum(0, np.arange(width) - radius)
    x1 = np.minimum(width, np.arange(width) + radius + 1)
    local_sum = (summed[y1[:, None], x1] - summed[y0[:, None], x1]
                 - summed[y1[:, None], x0] + summed[y0[:, None], x0])
    mean = local_sum / ((y1 - y0)[:, None] * (x1 - x0))
    return gray < mean * (1 - sensitivity)


def dilate(mask, radius=1):
    """정사각형 팽창. 이미지 밖은 배경이며 wrap-around하지 않는다."""
    if radius == 0:
        return mask.copy()
    height, width = mask.shape
    horizontal = np.pad(mask, ((0, 0), (radius, radius)))
    tmp = np.zeros_like(mask)
    for offset in range(2 * radius + 1):
        tmp |= horizontal[:, offset:offset + width]
    vertical = np.pad(tmp, ((radius, radius), (0, 0)))
    result = np.zeros_like(mask)
    for offset in range(2 * radius + 1):
        result |= vertical[offset:offset + height]
    return result


def connected_components(mask):
    """8방향 CCL. 행의 연속 구간과 union-find로 픽셀별 Python 순회를 줄인다."""
    height, width = mask.shape
    edges = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    starts = np.argwhere(edges == 1)
    ends = np.argwhere(edges == -1)
    parent = list(range(len(starts) + 1))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    runs, previous, current = [], [], []
    last_y = -1
    for run_id, ((y, left), (_, right)) in enumerate(zip(starts, ends), 1):
        y, left, right = int(y), int(left), int(right)
        if y != last_y:
            previous = current if y == last_y + 1 else []
            current, last_y = [], y
            cursor = 0
        while cursor < len(previous) and previous[cursor][1] < left:
            cursor += 1
        neighbor = cursor
        while neighbor < len(previous) and previous[neighbor][0] <= right:
            root_a, root_b = find(run_id), find(previous[neighbor][2])
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
            neighbor += 1
        current.append((left, right, run_id))
        runs.append((y, left, right, run_id))

    labels = np.zeros((height, width), dtype=np.int32)
    roots, boxes, areas = {}, [], []
    for y, left, right, run_id in runs:
        root = find(run_id)
        if root not in roots:
            roots[root] = len(boxes) + 1
            boxes.append([left, y, right, y + 1])
            areas.append(0)
        label = roots[root]
        labels[y, left:right] = label
        box = boxes[label - 1]
        box[0], box[1] = min(box[0], left), min(box[1], y)
        box[2], box[3] = max(box[2], right), max(box[3], y + 1)
        areas[label - 1] += right - left
    return labels, [Component(tuple(box), area, (i + 1,))
                    for i, (box, area) in enumerate(zip(boxes, areas))]


def merge_components(components, merge=True):
    components = [c for c in components if c.area >= 4]
    parent = list(range(len(components)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    if merge:
        for i, a in enumerate(components):
            ax0, ay0, ax1, ay1 = a.bbox
            for j in range(i + 1, len(components)):
                bx0, by0, bx1, by1 = components[j].bbox
                overlap = min(ax1, bx1) - max(ax0, bx0)
                gap = max(ay0, by0) - min(ay1, by1)
                if overlap >= 0.5 * min(ax1 - ax0, bx1 - bx0) and gap <= 0.3 * max(ay1 - ay0, by1 - by0):
                    parent[find(i)] = find(j)
    groups = {}
    for i, component in enumerate(components):
        groups.setdefault(find(i), []).append(component)
    return [Component(
        (min(c.bbox[0] for c in group), min(c.bbox[1] for c in group),
         max(c.bbox[2] for c in group), max(c.bbox[3] for c in group)),
        sum(c.area for c in group), tuple(m for c in group for m in c.members),
    ) for group in groups.values()]


def order_candidates(candidates):
    """모든 후보를 줄로 나눠 왼쪽부터 정렬한다. 개수 제한 전에 실행한다."""
    if not candidates:
        return []
    heights = sorted(c.bbox[3] - c.bbox[1] for c in candidates)
    median_height = heights[len(heights) // 2]
    lines = []
    for candidate in sorted(candidates, key=lambda c: ((c.bbox[1] + c.bbox[3]) / 2, c.bbox[0], c.members)):
        center = (candidate.bbox[1] + candidate.bbox[3]) / 2
        line = next((line for line in lines if abs(line[0] - center) < 0.6 * median_height), None)
        if line is None:
            line = [center, []]
            lines.append(line)
        line[1].append(candidate)
        line[0] = sum((c.bbox[1] + c.bbox[3]) / 2 for c in line[1]) / len(line[1])
    ordered = []
    for row, (_, items) in enumerate(sorted(lines, key=lambda line: line[0])):
        items.sort(key=lambda c: (c.bbox[0], c.bbox[1], c.members))
        for column, candidate in enumerate(items):
            space = column > 0 and candidate.bbox[0] - items[column - 1].bbox[2] > 0.7 * median_height
            ordered.append(replace(candidate, row=row, column=column, space_before=space))
    return ordered


def fit_to_28(ink):
    """면적 평균으로 긴 변 20px에 맞추고 무게중심을 (13.5, 13.5)로 이동한다."""
    ys, xs = np.nonzero(ink > 0.1)
    if not len(xs):
        return np.zeros((28, 28), dtype=np.float32)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    scale = 20 / max(x1 - x0, y1 - y0)
    target_w = max(1, int(np.floor((x1 - x0) * scale + 0.5)))
    target_h = max(1, int(np.floor((y1 - y0) * scale + 0.5)))

    def weights(origin, target, source):
        begin = origin + np.arange(target) / scale
        end = origin + (np.arange(target) + 1) / scale
        pixels = np.arange(source)
        return np.maximum(0, np.minimum(end[:, None], pixels + 1) - np.maximum(begin[:, None], pixels))

    wx, wy = weights(x0, target_w, ink.shape[1]), weights(y0, target_h, ink.shape[0])
    denominator = wy.sum(axis=1)[:, None] * wx.sum(axis=1)
    resized = np.divide(wy @ ink @ wx.T, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    image = np.zeros((28, 28), dtype=np.float32)
    ox, oy = (28 - target_w) // 2, (28 - target_h) // 2
    image[oy:oy + target_h, ox:ox + target_w] = np.clip(resized, 0, 1)
    mass = image.sum(dtype=np.float64)
    if mass == 0:
        return image
    yy, xx = np.indices(image.shape)
    dx = int(np.floor(13.5 - (image * xx).sum() / mass + 0.5))
    dy = int(np.floor(13.5 - (image * yy).sum() / mass + 0.5))
    out = np.zeros_like(image)
    sx0, sx1 = max(0, -dx), min(28, 28 - dx)
    sy0, sy1 = max(0, -dy), min(28, 28 - dy)
    out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = image[sy0:sy1, sx0:sx1]
    return out


def preprocess(image, config=None):
    """정답·라벨을 받지 않고 이미지에서 후보·CNN 입력만 생성한다."""
    config = config or Config()
    started = perf_counter()
    gray = grayscale(image)
    if config.background == "black":
        gray = 1 - gray
    height, width = gray.shape
    mask = binarize(gray, config.binarization, config.sensitivity)
    labels, components = connected_components(dilate(mask) if config.merge else mask)
    candidates, dropped = [], []
    for component in merge_components(components, config.merge):
        x0, y0, x1, y1 = component.bbox
        w, h = x1 - x0, y1 - y0
        reason = None
        if h < config.min_height:
            reason = "small" if component.area >= 20 else "dot"
        elif x0 == 0 or y0 == 0 or x1 == width or y1 == height:
            reason = "border"
        elif h > 0.9 * height or w > 0.6 * width:
            reason = "large"
        elif w > 1.8 * h:
            reason = "wide"
        elif component.area < 0.04 * w * h:
            reason = "sparse"
        if reason:
            dropped.append({**asdict(component), "reason": reason})
            continue
        support = mask[y0:y1, x0:x1] & np.isin(labels[y0:y1, x0:x1], component.members)
        ys, xs = np.nonzero(support)
        if not len(xs):
            continue
        bbox = (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
        p = config.crop_padding
        crop_bbox = (max(0, bbox[0] - p), max(0, bbox[1] - p), min(width, bbox[2] + p), min(height, bbox[3] + p))
        candidates.append(Candidate(bbox, component.bbox, crop_bbox, int(support.sum()), component.members))
    ordered = order_candidates(candidates)
    kept, excess = ordered[:config.max_digits], ordered[config.max_digits:]
    inputs = np.zeros((len(kept), 1, 28, 28), dtype=np.float32)
    for i, candidate in enumerate(kept):
        x0, y0, x1, y1 = candidate.crop_bbox
        local_labels = labels[y0:y1, x0:x1]
        own = np.isin(local_labels, candidate.members)
        if config.crop_mode == "mask":
            crop = (mask[y0:y1, x0:x1] & own).astype(np.float64)
        else:
            # 원본 회색조 보존. 다른 덩어리만 제거하고 미검출된 옅은 획은 유지.
            crop = np.where(own | (local_labels == 0), 1 - gray[y0:y1, x0:x1], 0)
        inputs[i, 0] = fit_to_28(crop)
    return Result(inputs, kept, excess, dropped, mask, len(components), (perf_counter() - started) * 1000)


def save_result(image, result, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "inputs.npy", result.inputs, allow_pickle=False)
    Image.fromarray(np.where(result.mask, 0, 255).astype(np.uint8)).save(output / "mask.png")
    overlay = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for color, items in (("red", result.candidates), ("orange", result.excess)):
        for index, candidate in enumerate(items):
            x0, y0, x1, y1 = candidate.bbox
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color)
            draw.text((x0, max(0, y0 - 12)), f"{index}:r{candidate.row}c{candidate.column}", fill=color)
    overlay.save(output / "overlay.png")
    sheet = Image.new("L", (4 * 112, max(1, (len(result.inputs) + 3) // 4) * 132), "white")
    labels = ImageDraw.Draw(sheet)
    for index, crop in enumerate(result.inputs[:, 0]):
        x, y = index % 4 * 112, index // 4 * 132
        tile = Image.fromarray(np.floor(crop * 255 + 0.5).astype(np.uint8)).resize((112, 112), Image.Resampling.NEAREST)
        sheet.paste(tile, (x, y + 20))
        labels.text((x + 2, y + 2), str(index), fill="black")
    sheet.save(output / "crops.png")
    report = {
        "config": asdict(config), "input_shape": list(result.inputs.shape),
        "input_format": "NCHW float32 0..1; black background, bright grayscale ink",
        "bbox_format": "xyxy, right/bottom exclusive; undilated detected ink",
        "candidates": [asdict(c) for c in result.candidates],
        "excess": [asdict(c) for c in result.excess], "overflow": result.overflow,
        "dropped": result.dropped, "component_count": result.component_count,
        "elapsed_ms": result.elapsed_ms, "classification": "not performed",
    }
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "build" / "preprocessing" / "single")
    parser.add_argument("--binarization", choices=("adaptive", "otsu"), default="adaptive")
    parser.add_argument("--sensitivity", type=float, default=0.2)
    parser.add_argument("--min-height", type=int, default=12)
    parser.add_argument("--max-digits", type=int, default=16)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--crop-mode", choices=("grayscale", "mask"), default="grayscale")
    args = parser.parse_args()
    config = Config(args.binarization, args.sensitivity, args.min_height, not args.no_merge,
                    args.max_digits, args.background, crop_mode=args.crop_mode)
    if args.output_dir.exists():
        parser.error(f"output already exists: {args.output_dir}")
    image = load_grayscale(args.image)
    result = preprocess(image, config)
    save_result(image, result, config, args.output_dir)
    print(f"{len(result.candidates)} candidates; overflow={result.overflow}; {result.elapsed_ms:.1f} ms; {args.output_dir}")


if __name__ == "__main__":
    main()
