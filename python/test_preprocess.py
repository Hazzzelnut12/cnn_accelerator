"""위치 탐지·회색조·정렬·초과 처리의 회귀 검사. unittest 표준 라이브러리 사용."""

from collections import deque
import unittest

import numpy as np

from preprocess import Config, binarize, connected_components, fit_to_28, preprocess


def flood_fill(mask):
    """CCL 구현과 다른 BFS 방식으로 8방향 연결 정답을 만든다."""
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x]:
                continue
            count += 1
            labels[y, x] = count
            pending = deque([(y, x)])
            while pending:
                cy, cx = pending.popleft()
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = count
                            pending.append((ny, nx))
    return labels


class PreprocessingTests(unittest.TestCase):
    def test_ccl_matches_independent_bfs(self):
        rng = np.random.default_rng(31)
        for density in (0.0, 0.05, 0.3, 0.8, 1.0):
            for _ in range(6):
                mask = rng.random((23, 37)) < density
                actual, components = connected_components(mask)
                expected = flood_fill(mask)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(len(components), int(expected.max()))
                self.assertEqual(sum(c.area for c in components), int(mask.sum()))
                for label, component in enumerate(components, 1):
                    ys, xs = np.nonzero(expected == label)
                    self.assertEqual(component.bbox, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    def test_diagonal_pixels_are_connected(self):
        labels, components = connected_components(np.eye(7, dtype=bool))
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].area, 7)
        self.assertEqual(int(labels.max()), 1)

    def test_otsu_includes_zero_threshold_black_ink(self):
        gray = np.ones((64, 64))
        gray[20:40, 25:35] = 0
        expected = gray == 0
        np.testing.assert_array_equal(binarize(gray, "otsu"), expected)

    def test_uniform_inputs_produce_no_otsu_candidates(self):
        for value in (0, 127, 255):
            result = preprocess(np.full((64, 64), value, dtype=np.uint8), Config(binarization="otsu"))
            self.assertEqual(result.inputs.shape, (0, 1, 28, 28))
            self.assertFalse(result.overflow)

    def test_adaptive_matches_local_window_definition(self):
        rng = np.random.default_rng(7)
        gray = rng.random((19, 29))
        expected = np.zeros(gray.shape, dtype=bool)
        for y in range(19):
            for x in range(29):
                local = gray[max(0, y - 7):min(19, y + 8), max(0, x - 7):min(29, x + 8)]
                expected[y, x] = gray[y, x] < local.mean() * 0.8
        np.testing.assert_array_equal(binarize(gray), expected)

    def test_grayscale_is_not_replaced_by_binary_mask(self):
        image = np.full((80, 100), 255, dtype=np.uint8)
        image[20:40, 25:35] = 64
        gray = preprocess(image, Config(binarization="otsu"))
        mask = preprocess(image, Config(binarization="otsu", crop_mode="mask"))
        self.assertEqual(len(gray.candidates), 1)
        self.assertEqual(gray.candidates[0].bbox, (25, 20, 35, 40))
        self.assertEqual(gray.candidates[0].component_bbox, (24, 19, 36, 41))
        self.assertEqual(gray.candidates[0].crop_bbox, (23, 18, 37, 42))
        self.assertAlmostEqual(float(gray.inputs.max()), 191 / 255, places=6)
        self.assertEqual(float(mask.inputs.max()), 1.0)
        self.assertEqual(gray.inputs.dtype, np.float32)

    def test_opposite_backgrounds_produce_identical_inputs(self):
        image = np.full((80, 100), 255, dtype=np.uint8)
        image[20:40, 25:35] = 64
        white = preprocess(image)
        black = preprocess(255 - image, Config(background="black"))
        np.testing.assert_array_equal(white.mask, black.mask)
        np.testing.assert_array_equal(white.inputs, black.inputs)
        self.assertEqual(white.candidates, black.candidates)

    def test_broken_vertical_strokes_merge(self):
        image = np.full((90, 100), 255, dtype=np.uint8)
        image[15:21, 25:33] = 0
        image[24:38, 25:33] = 0
        merged = preprocess(image)
        separate = preprocess(image, Config(merge=False, min_height=4))
        self.assertEqual(len(merged.candidates), 1)
        self.assertEqual(merged.candidates[0].bbox, (25, 15, 33, 38))
        self.assertEqual(len(separate.candidates), 2)

    def test_overflow_selects_reading_order_not_largest_area(self):
        image = np.full((360, 320), 255, dtype=np.uint8)
        boxes = []
        for row in range(4):
            for col in range(5):
                # 뒤쪽 줄일수록 크게 만들어 면적순 선택과 구분한다.
                x, y = 20 + col * 55, 20 + row * 80
                width, height = 6 + row * 2, 15 + row * 2
                image[y:y + height, x:x + width] = 0
                boxes.append((x, y, x + width, y + height))
        result = preprocess(image)
        self.assertEqual([c.bbox for c in result.candidates], boxes[:16])
        self.assertEqual([c.bbox for c in result.excess], boxes[16:])
        self.assertTrue(result.overflow)
        self.assertEqual(result.inputs.shape, (16, 1, 28, 28))
        self.assertEqual([(c.row, c.column) for c in result.candidates], [(r, c) for r in range(4) for c in range(5)][:16])

    def test_boundary_is_filtered_without_wrapping(self):
        image = np.full((100, 100), 255, dtype=np.uint8)
        image[0:20, 0:7] = 0
        image[40:60, 60:70] = 0
        result = preprocess(image)
        self.assertEqual([c.bbox for c in result.candidates], [(60, 40, 70, 60)])
        self.assertIn("border", [d["reason"] for d in result.dropped])

    def test_normalization_size_center_and_no_wrapping(self):
        ink = np.zeros((37, 29))
        ink[5:35, 2:17] = 0.7
        normalized = fit_to_28(ink)
        ys, xs = np.nonzero(normalized)
        self.assertEqual((ys.max() - ys.min() + 1, xs.max() - xs.min() + 1), (20, 10))
        yy, xx = np.indices(normalized.shape)
        self.assertLessEqual(abs(float((normalized * xx).sum() / normalized.sum()) - 13.5), 0.5)
        self.assertLessEqual(abs(float((normalized * yy).sum() / normalized.sum()) - 13.5), 0.5)
        self.assertGreaterEqual(float(normalized.min()), 0)
        self.assertLessEqual(float(normalized.max()), 1)
        self.assertTrue(np.all(fit_to_28(np.zeros((20, 20))) == 0))

    def test_invalid_values_are_rejected(self):
        for image in (np.zeros((10, 10, 3)), np.array([[np.nan]]), np.array([[256.0]])):
            with self.assertRaises(ValueError):
                preprocess(image)
        for params in ({"max_digits": 17}, {"min_height": 0}, {"sensitivity": 1}):
            with self.assertRaises(ValueError):
                Config(**params)


if __name__ == "__main__":
    unittest.main()
