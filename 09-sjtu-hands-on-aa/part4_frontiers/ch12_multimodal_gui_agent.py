#!/usr/bin/env python3
"""
Chapter 12: Multimodal & GUI Agents (多模态与图形界面GUI智能体)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 4: Frontiers

Implements:
1. Visual Grounding Coordinate Mapper (`VisualGroundingMapper`):
   - Converts normalized [0, 1000] VLM bounding boxes `[ymin, xmin, ymax, xmax]`
     to screen pixel coordinates `(px_x, px_y)` and computes Intersection-over-Union (IoU)
2. Set-of-Mark (SoM) Accessibility / UI Element Tree (`UIElement`, `MockGUIEnvironment`)
3. GUI Action Space Executor (`CLICK`, `TYPE`, `SCROLL`, `FINISH`) completing an end-to-end
   flight-search web form workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BoundingBox:
    """Normalized [0, 1000] bounding box (ymin, xmin, ymax, xmax)."""
    ymin: float
    xmin: float
    ymax: float
    xmax: float

    def center_normalized(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def to_pixel_center(self, screen_w: int = 1920, screen_h: int = 1080) -> Tuple[int, int]:
        cx, cy = self.center_normalized()
        px = int(round((cx / 1000.0) * screen_w))
        py = int(round((cy / 1000.0) * screen_h))
        return px, py

    def contains_pixel(self, px: int, py: int, screen_w: int = 1920, screen_h: int = 1080) -> bool:
        nx = (px / screen_w) * 1000.0
        ny = (py / screen_h) * 1000.0
        return (self.xmin <= nx <= self.xmax) and (self.ymin <= ny <= self.ymax)

    def iou(self, other: "BoundingBox") -> float:
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymax = min(self.ymax, other.ymax)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area_self = max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)
        area_other = max(0.0, other.xmax - other.xmin) * max(0.0, other.ymax - other.ymin)
        union_area = area_self + area_other - inter_area
        return round(inter_area / union_area, 4) if union_area > 0 else 0.0


@dataclass
class UIElement:
    som_id: int
    role: str
    label: str
    bbox: BoundingBox
    value: str = ""
    visible: bool = True


@dataclass(frozen=True)
class GUIAction:
    action_type: str  # "CLICK" | "TYPE" | "SCROLL" | "FINISH"
    pixel_xy: Optional[Tuple[int, int]] = None
    text: str = ""
    scroll_dy: int = 0


class MockGUIEnvironment:
    """Simulates a 1920x1080 desktop browser GUI with Set-of-Mark (SoM) grounded elements."""

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.focused_som_id: Optional[int] = None
        self.scroll_offset_y: int = 0
        self.submitted: bool = False
        self.elements: Dict[int, UIElement] = {
            1: UIElement(1, "input", "Origin City", BoundingBox(150, 200, 210, 500)),
            2: UIElement(2, "input", "Destination City", BoundingBox(150, 550, 210, 850)),
            3: UIElement(3, "button", "Search Flights", BoundingBox(400, 400, 470, 600), visible=False),
        }

    def step(self, action: GUIAction) -> Dict[str, Any]:
        if action.action_type == "SCROLL":
            self.scroll_offset_y += action.scroll_dy
            if self.scroll_offset_y >= 200:
                self.elements[3].visible = True
            return {"status": "scrolled", "scroll_y": self.scroll_offset_y}

        if action.action_type == "CLICK" and action.pixel_xy is not None:
            px, py = action.pixel_xy
            for som_id, elem in self.elements.items():
                if elem.visible and elem.bbox.contains_pixel(px, py, self.screen_w, self.screen_h):
                    self.focused_som_id = som_id
                    if elem.role == "button" and elem.label == "Search Flights":
                        self.submitted = True
                    return {"status": "clicked", "som_id": som_id, "label": elem.label}
            return {"status": "miss", "pixel_xy": action.pixel_xy}

        if action.action_type == "TYPE":
            if self.focused_som_id is not None and self.focused_som_id in self.elements:
                elem = self.elements[self.focused_som_id]
                elem.value = action.text
                return {"status": "typed", "som_id": elem.som_id, "value": elem.value}
            return {"status": "error", "reason": "No input element focused"}

        if action.action_type == "FINISH":
            success = (
                self.elements[1].value == "Shanghai"
                and self.elements[2].value == "Boston"
                and self.submitted
            )
            return {"status": "finished", "task_success": success}

        return {"status": "unknown_action"}


def run_chapter_demo() -> Dict[str, Any]:
    env = MockGUIEnvironment()
    pred_box = BoundingBox(155, 205, 205, 495)
    gt_box = env.elements[1].bbox
    iou_score = pred_box.iou(gt_box)

    # Execute grounded GUI agent trajectory:
    # 1. Click Origin input -> Type "Shanghai"
    # 2. Click Destination input -> Type "Boston"
    # 3. Scroll down to reveal Search button -> Click Search -> Finish
    p1 = env.elements[1].bbox.to_pixel_center()
    p2 = env.elements[2].bbox.to_pixel_center()
    p3 = env.elements[3].bbox.to_pixel_center()

    actions = [
        GUIAction("CLICK", pixel_xy=p1),
        GUIAction("TYPE", text="Shanghai"),
        GUIAction("CLICK", pixel_xy=p2),
        GUIAction("TYPE", text="Boston"),
        GUIAction("SCROLL", scroll_dy=250),
        GUIAction("CLICK", pixel_xy=p3),
        GUIAction("FINISH"),
    ]

    logs = [env.step(a) for a in actions]
    return {
        "grounding_iou": iou_score,
        "origin_pixel_center": p1,
        "trajectory_length": len(logs),
        "final_state": logs[-1],
        "task_success": bool(logs[-1].get("task_success", False)),
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 12 Demo] Grounding IoU={res['grounding_iou']}, PixelCenter={res['origin_pixel_center']}, GUI Task Success={res['task_success']}")
