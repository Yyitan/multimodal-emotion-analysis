"""Auditable mapping from sequence positions to original-video intervals."""
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence
import math


@dataclass(frozen=True)
class TimeMapping:
    start_sec: Optional[float]
    end_sec: Optional[float]
    keyframe_sec: Optional[float]
    method: str
    confidence: str
    approximate: bool
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return {"start_sec": self.start_sec, "end_sec": self.end_sec,
                "keyframe_sec": self.keyframe_sec, "method": self.method,
                "confidence": self.confidence, "approximate": self.approximate,
                "note": self.note}


def _parse_alignment(alignment: Sequence[Any]) -> Dict[int, Dict[str, Any]]:
    parsed = {}
    for default_position, item in enumerate(alignment):
        if isinstance(item, Mapping):
            position = int(item.get("position", default_position))
            start = float(item["start_sec"]); end = float(item["end_sec"])
            confidence = str(item.get("confidence", "aligned"))
        else:
            position = default_position; start, end = float(item[0]), float(item[1])
            confidence = "aligned"
        if not (math.isfinite(start) and math.isfinite(end) and 0.0 <= start <= end):
            raise ValueError("invalid forced-alignment interval at position %d" % position)
        parsed[position] = {"start_sec": start, "end_sec": end, "confidence": confidence}
    return parsed


def map_position_window(start_index: int, end_index: int, total_positions: int = 50,
                        duration_sec: Optional[float] = None,
                        forced_alignment: Optional[Sequence[Any]] = None,
                        include_keyframe: bool = False) -> TimeMapping:
    """Map half-open positions [start,end), preferring complete forced alignment.

    Uniform fallback is always marked ``approximate``. Without alignment or a
    measured duration, timestamps remain null rather than being fabricated.
    """
    start_index, end_index, total_positions = int(start_index), int(end_index), int(total_positions)
    if total_positions <= 0 or not (0 <= start_index < end_index <= total_positions):
        raise ValueError("invalid position window")
    if forced_alignment is not None:
        parsed = _parse_alignment(forced_alignment)
        selected = [parsed.get(i) for i in range(start_index, end_index)]
        if all(x is not None for x in selected):
            start = min(x["start_sec"] for x in selected); end = max(x["end_sec"] for x in selected)
            keyframe = (start + end) / 2.0 if include_keyframe else None
            confidences = sorted(set(x["confidence"] for x in selected))
            return TimeMapping(start, end, keyframe, "forced_alignment", "+".join(confidences),
                               False, "由提供的强制对齐区间聚合；未进行时间猜测。")
    if duration_sec is not None:
        duration = float(duration_sec)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_sec must be positive and finite")
        start = duration * start_index / total_positions; end = duration * end_index / total_positions
        keyframe = (start + end) / 2.0 if include_keyframe else None
        method = "uniform_50_position" if total_positions == 50 else "uniform_position"
        return TimeMapping(start, end, keyframe, method, "approximate", True,
                           "缺少完整强制对齐，按序列位置占视频总时长的比例均匀映射，仅供近似回看。")
    return TimeMapping(None, None, None, "unavailable", "unavailable", False,
                       "未提供完整强制对齐或视频时长，故不输出时间戳。")
