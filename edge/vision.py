from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised in runtime environments without Pillow
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None


@dataclass(frozen=True)
class ImageFeatureEstimate:
    center_x: float
    center_y: float
    grip_width: float
    bbox: tuple[float, float, float, float]
    component_area: int

    def to_dict(self) -> dict[str, object]:
        left, top, right, bottom = self.bbox
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "grip_width": self.grip_width,
            "bbox": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            },
            "component_area": self.component_area,
        }


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_saturation(red: int, green: int, blue: int) -> float:
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    if maximum <= 0:
        return 0.0
    return (maximum - minimum) / maximum


def resize_for_analysis(image: Image.Image, *, max_dimension: int = 192) -> Image.Image:
    width, height = image.size
    largest_dimension = max(width, height)
    if largest_dimension <= max_dimension:
        return image.copy()

    scale = max_dimension / float(largest_dimension)
    resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(resized_size)


def build_foreground_mask(image: Image.Image) -> tuple[list[list[bool]], int]:
    width, height = image.size
    crop_top = int(height * 0.42)
    pixels = image.load()
    mask: list[list[bool]] = [[False for _ in range(width)] for _ in range(height - crop_top)]

    for y in range(crop_top, height):
        mask_row = mask[y - crop_top]
        y_ratio = y / max(height - 1, 1)
        for x in range(width):
            red, green, blue = pixels[x, y]
            brightness = max(red, green, blue) / 255.0
            saturation = compute_saturation(red, green, blue)

            is_colorful = saturation >= 0.22 and brightness >= 0.16
            is_bright_lower_area = y_ratio >= 0.62 and brightness >= 0.32 and saturation >= 0.10
            mask_row[x] = is_colorful or is_bright_lower_area

    return mask, crop_top


def component_score(
    *,
    area: int,
    width: int,
    height: int,
    mean_y_ratio: float,
    touches_side_edge: bool,
) -> float:
    score = float(area) * (0.8 + mean_y_ratio)

    if height > width * 2.4:
        score *= 0.18
    if touches_side_edge and height > width * 1.2:
        score *= 0.28
    if width < 6 or height < 6:
        score *= 0.4

    return score


def find_best_component(mask: list[list[bool]]) -> tuple[int, int, int, int, float, float, int]:
    if not mask or not mask[0]:
        raise RuntimeError("Image crop is empty; cannot estimate centroid.")

    height = len(mask)
    width = len(mask[0])
    visited = [[False for _ in range(width)] for _ in range(height)]
    best_component: tuple[int, int, int, int, float, float, int] | None = None
    best_score = -1.0

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y][start_x] or visited[start_y][start_x]:
                continue

            queue = deque([(start_x, start_y)])
            visited[start_y][start_x] = True

            min_x = max_x = start_x
            min_y = max_y = start_y
            area = 0
            weighted_x = 0.0
            weighted_y = 0.0

            while queue:
                x, y = queue.popleft()
                area += 1
                weighted_x += x
                weighted_y += y
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        if delta_x == 0 and delta_y == 0:
                            continue
                        next_x = x + delta_x
                        next_y = y + delta_y
                        if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                            continue
                        if visited[next_y][next_x] or not mask[next_y][next_x]:
                            continue
                        visited[next_y][next_x] = True
                        queue.append((next_x, next_y))

            component_width = max_x - min_x + 1
            component_height = max_y - min_y + 1
            mean_x = weighted_x / area
            mean_y = weighted_y / area
            score = component_score(
                area=area,
                width=component_width,
                height=component_height,
                mean_y_ratio=mean_y / max(height - 1, 1),
                touches_side_edge=min_x == 0 or max_x == width - 1,
            )

            if score > best_score:
                best_score = score
                best_component = (min_x, min_y, max_x, max_y, mean_x, mean_y, area)

    if best_component is None:
        raise RuntimeError("Could not find a colorful foreground component for centroid estimation.")

    return best_component


def estimate_image_features(image_path: str | Path) -> ImageFeatureEstimate:
    if Image is None:
        raise RuntimeError(f"Pillow is required for image feature extraction: {PIL_IMPORT_ERROR}")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image was not found: {path}")

    with Image.open(path) as opened:
        analyzed = resize_for_analysis(opened.convert("RGB"))

    width, height = analyzed.size
    mask, crop_top = build_foreground_mask(analyzed)
    min_x, min_y, max_x, max_y, mean_x, mean_y, area = find_best_component(mask)

    center_x = clamp(mean_x / max(width - 1, 1), 0.0, 1.0)
    center_y = clamp((crop_top + mean_y) / max(height - 1, 1), 0.0, 1.0)

    component_width = max_x - min_x + 1
    # Width is converted into the training-time grip-width scale heuristically.
    grip_width = clamp((component_width / max(width, 1)) * 3.2, 0.30, 0.95)

    return ImageFeatureEstimate(
        center_x=round(center_x, 4),
        center_y=round(center_y, 4),
        grip_width=round(grip_width, 4),
        bbox=(
            round(min_x / max(width - 1, 1), 4),
            round((crop_top + min_y) / max(height - 1, 1), 4),
            round(max_x / max(width - 1, 1), 4),
            round((crop_top + max_y) / max(height - 1, 1), 4),
        ),
        component_area=area,
    )


def render_estimate_overlay(
    image_path: str | Path,
    output_path: str | Path,
    *,
    estimate: ImageFeatureEstimate,
    title_lines: list[str] | None = None,
) -> Path:
    if Image is None or ImageDraw is None:
        raise RuntimeError(f"Pillow is required for image rendering: {PIL_IMPORT_ERROR}")

    input_path = Path(image_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as opened:
        canvas = opened.convert("RGB")

    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default() if ImageFont is not None else None

    left = int(estimate.bbox[0] * width)
    top = int(estimate.bbox[1] * height)
    right = int(estimate.bbox[2] * width)
    bottom = int(estimate.bbox[3] * height)
    center_x = int(estimate.center_x * width)
    center_y = int(estimate.center_y * height)

    box_color = (255, 82, 82)
    center_color = (255, 232, 86)
    line_width = max(2, width // 320)

    draw.rectangle((left, top, right, bottom), outline=box_color, width=line_width)
    draw.ellipse(
        (
            center_x - 7,
            center_y - 7,
            center_x + 7,
            center_y + 7,
        ),
        fill=center_color,
        outline=(24, 24, 24),
        width=2,
    )
    draw.line((center_x - 18, center_y, center_x + 18, center_y), fill=center_color, width=2)
    draw.line((center_x, center_y - 18, center_x, center_y + 18), fill=center_color, width=2)

    lines = title_lines or []
    lines.extend(
        [
            f"center=({estimate.center_x:.3f}, {estimate.center_y:.3f})",
            f"grip_width={estimate.grip_width:.3f}",
        ]
    )

    if lines:
        line_height = 16
        panel_width = min(width - 20, 360)
        panel_height = 14 + line_height * len(lines)
        panel_box = (10, 10, 10 + panel_width, 10 + panel_height)
        draw.rounded_rectangle(panel_box, radius=12, fill=(18, 22, 28), outline=(255, 255, 255))
        text_y = 18
        for line in lines:
            draw.text((18, text_y), line, fill=(245, 245, 245), font=font)
            text_y += line_height

    canvas.save(destination, quality=95)
    return destination
