# Copyright (c) 2026, Manfred Moitzi
# License: MIT License
"""Convert :class:`Path` objects into polyline vertices with bulge values.

Cubic Bèzier-curve segments which represent an exact circular arc are converted
to polyline arc segments defined by a bulge value instead of being flattened
into many short line segments. Adjacent arc segments located on the same circle
are merged into a single arc segment, and a full circle is split into two
semicircles. Free-form curves which do not represent a circular arc are still
flattened into line segments and the approximation error is controlled by the
`distance` and `segments` arguments as in :meth:`Path.flattening`.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence

from ezdxf.math import Bezier3P, Bezier4P, Vec2, Vec3

from .commands import Command
from .path import Path

__all__ = ["vertices_and_bulges"]

# Relative tolerance for the circular-arc detection.  The cubic Bèzier curves
# created by ezdxf for circular arcs follow an exact construction rule, so the
# remaining deviation from this rule is floating point noise.
ARC_TOL = 1e-9
# Bulge values below this threshold are treated as straight line segments,
# matching the rule in tools.add_2d_polyline() (see issue #329).
MIN_BULGE = 1e-6
# A closed arc run whose total angle reaches a full circle is split into two
# semicircles.
FULL_CIRCLE = math.tau


class _Arc(NamedTuple):
    start: Vec3
    end: Vec3
    center: Vec2
    radius: float
    angle: float  # signed arc angle in radians, direction is included


class _Line(NamedTuple):
    start: Vec3
    end: Vec3


class _Curve(NamedTuple):
    points: list[Vec3]


def _circular_arc(
    p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2
) -> Optional[tuple[Vec2, float, float]]:
    """Return the (center, radius, signed angle) of a circular arc represented
    by the cubic Bèzier-curve with control points p0..p3, or ``None`` if the
    curve does not represent a circular arc.

    Circular arcs created by :func:`cubic_bezier_arc_parameters` have equal
    tangent lengths at both ends and both tangent vectors are perpendicular to
    the radius vectors of the same circle.

    """
    tangent1 = p1 - p0
    tangent2 = p3 - p2
    length1 = tangent1.magnitude
    length2 = tangent2.magnitude
    if length1 < 1e-12 or length2 < 1e-12:
        return None
    # equal tangent lengths at both endpoints
    if not math.isclose(length1, length2, rel_tol=ARC_TOL):
        return None
    chord = p3 - p0
    if chord.magnitude < 1e-12:
        return None

    normal1 = Vec2(-tangent1.y, tangent1.x)
    normal2 = Vec2(-tangent2.y, tangent2.x)
    det = normal1.det(normal2)
    if abs(det) < 1e-12:
        # Parallel normals: the arc is a semicircle and its center is the
        # midpoint of the chord.
        if not math.isclose(
            (tangent1 + tangent2).magnitude, 0.0, abs_tol=ARC_TOL * length1
        ):
            return None
        center = p0.lerp(p3)
    else:
        # Intersection of the radial lines through p0 and p3:
        # p0 + s * normal1 = p3 + u * normal2
        s = chord.det(normal2) / det
        center = p0 + normal1 * s

    radius = center.distance(p0)
    if radius < 1e-12:
        return None
    if not math.isclose(radius, center.distance(p3), rel_tol=ARC_TOL):
        return None

    start_angle = (p0 - center).angle
    end_angle = (p3 - center).angle
    # The sign of the angle follows the arc orientation.  The midpoint of the
    # Bèzier curve (t = 0.5) lies on the arc and its side relative to the chord
    # distinguishes the orientation even for a semicircle, where the center is
    # located directly on the chord: a counter-clockwise arc lies to the left
    # of the chord (positive), a clockwise arc to the right (negative).
    midpoint = (p0 + 3.0 * p1 + 3.0 * p2 + p3) / 8.0
    if (midpoint - p0).det(chord) >= 0.0:
        angle = (end_angle - start_angle) % math.tau
    else:
        angle = -((start_angle - end_angle) % math.tau)

    # Construction rule of the circular-arc Bèzier curves:
    # tangent length = 4/3 * radius * tan(angle / 4)
    expected_length = 4.0 / 3.0 * radius * abs(math.tan(angle / 4.0))
    if not math.isclose(length1, expected_length, rel_tol=ARC_TOL):
        return None
    return center, radius, angle


def _collect_segments(path: Path, distance: float, segments: int) -> list:
    """Split a single path into line segments, circular arcs and free-form
    curves represented by flattened line segments."""
    result: list = []
    start = Vec3(path.start)
    for cmd in path:
        end = Vec3(cmd.end)
        if cmd.type == Command.LINE_TO or cmd.type == Command.MOVE_TO:
            result.append(_Line(start, end))
        elif cmd.type == Command.CURVE4_TO:
            arc = _circular_arc(
                Vec2(start), Vec2(cmd.ctrl1), Vec2(cmd.ctrl2), Vec2(end)
            )
            if arc is not None:
                center, radius, angle = arc
                result.append(_Arc(start, end, center, radius, angle))
            else:
                points = list(
                    Bezier4P((start, Vec3(cmd.ctrl1), Vec3(cmd.ctrl2), end)).flattening(
                        distance, segments
                    )
                )
                result.append(_Curve(points))
        elif cmd.type == Command.CURVE3_TO:
            # A quadratic Bèzier-curve can not represent a circular arc,
            # it is always a parabola and has to be flattened.
            points = list(
                Bezier3P((start, Vec3(cmd.ctrl), end)).flattening(distance, segments)
            )
            result.append(_Curve(points))
        else:
            raise ValueError(f"invalid path command: {cmd.type}")
        start = end
    return result


def _merge_arcs(segments: Sequence) -> list:
    """Merge adjacent circular arcs located on the same circle into a single
    arc segment."""
    merged: list = []
    for segment in segments:
        if isinstance(segment, _Arc) and merged and isinstance(merged[-1], _Arc):
            previous = merged[-1]
            tol = ARC_TOL * max(previous.radius, 1.0)
            if (
                previous.end.isclose(segment.start, abs_tol=tol)
                and previous.center.isclose(segment.center, abs_tol=tol)
                and math.isclose(previous.radius, segment.radius, rel_tol=ARC_TOL)
                and math.copysign(1.0, previous.angle)
                == math.copysign(1.0, segment.angle)
            ):
                merged[-1] = _Arc(
                    previous.start,
                    segment.end,
                    previous.center,
                    previous.radius,
                    previous.angle + segment.angle,
                )
                continue
        merged.append(segment)
    return merged


def _emit_curve(vertices: list[Vec3], bulges: list[float], points: list[Vec3]) -> None:
    # The first point is coincident with the current end vertex and is skipped.
    if not vertices:
        vertices.extend(points)
        bulges.extend([0.0] * len(points))
        return
    for point in points[1:]:
        if not vertices[-1].isclose(point):
            vertices.append(point)
            bulges.append(0.0)


def vertices_and_bulges(
    path: Path, distance: float, segments: int
) -> tuple[list[Vec3], list[float], bool]:
    """Convert a single (non multi-) :class:`Path` into polyline vertices and
    bulge values.

    Args:
        path: a single :class:`Path` object without sub-paths
        distance: maximum flattening distance for free-form curves
        segments: minimum segment count per free-form Bèzier curve

    Returns:
        Tuple of (vertices, bulges, closed).  The bulges list has the same
        length as the vertices list; the bulge value of a vertex defines the
        arc from that vertex to the next vertex.  A closed path does not repeat
        its start vertex at the end, the closed state is returned separately.

    """
    segments_list = _merge_arcs(_collect_segments(path, distance, segments))
    if not segments_list:
        if len(path):
            return [Vec3(path.start)], [0.0], False
        return [], [], False

    # A path is only closed if it contains an explicit closing segment, i.e.
    # its last command ends at the very first vertex.  An open path whose end
    # merely coincides with its start keeps the repeated vertex and is not
    # flagged as closed.
    start_point = Vec3(path.start)
    closed = bool(path.command_codes()) and Vec3(path.end).isclose(start_point)
    vertices: list[Vec3] = []
    bulges: list[float] = []
    # Set while a full-circle arc is split into two semicircles, whose closing
    # semicircle already carries its bulge on the midpoint vertex and must not
    # be processed by the generic closed-path handling at the end.
    full_circle_handled = False

    def add_vertex(point: Vec3, bulge: float = 0.0) -> None:
        if vertices and vertices[-1].isclose(point):
            return
        vertices.append(point)
        bulges.append(bulge)

    def add_arc(start: Vec3, end: Vec3, angle: float) -> None:
        if not vertices:
            vertices.append(start)
            bulges.append(0.0)
        bulge = math.tan(angle / 4.0)
        if abs(bulge) < MIN_BULGE:
            bulge = 0.0
        # The bulge value belongs to the start vertex of the arc segment.
        bulges[-1] = bulge
        add_vertex(end)

    for segment in segments_list:
        if isinstance(segment, _Line):
            if not vertices:
                add_vertex(segment.start)
            add_vertex(segment.end)
        elif isinstance(segment, _Curve):
            _emit_curve(vertices, bulges, segment.points)
        elif isinstance(segment, _Arc):
            if (
                closed
                and math.isclose(abs(segment.angle), FULL_CIRCLE, abs_tol=ARC_TOL)
                and segment.start.isclose(segment.end)
            ):
                # A full circle can not be represented by a single bulge value
                # (start and end points would be coincident), therefore it is
                # split into two semicircles.  Both semicircles carry a bulge of
                # +/- 1; the second one connects back to the start vertex.
                direction = math.copysign(1.0, segment.angle)
                start_angle = (Vec2(segment.start) - segment.center).angle
                midpoint2d = segment.center + Vec2.from_angle(
                    start_angle + direction * math.pi, segment.radius
                )
                midpoint = Vec3(midpoint2d.x, midpoint2d.y, segment.start.z)
                if not vertices:
                    vertices.append(segment.start)
                    bulges.append(direction)  # first semicircle
                else:
                    bulges[-1] = direction
                vertices.append(midpoint)
                bulges.append(direction)  # closing semicircle (at midpoint)
                full_circle_handled = True
            else:
                add_arc(segment.start, segment.end, segment.angle)
        else:  # pragma: no cover
            raise TypeError(type(segment))

    if (
        closed
        and not full_circle_handled
        and vertices
        and vertices[-1].isclose(vertices[0])
    ):
        # A closed path uses the closed flag instead of a repeated start
        # vertex.  The repeated closing vertex only carries a bulge when the
        # closing segment itself is an arc.  When the closing segment is a
        # straight line its bulge is zero and an existing bulge on the
        # previous vertex (e.g. the second semicircle of a circle) is kept.
        closing_bulge = bulges.pop()
        vertices.pop()
        if closing_bulge:
            bulges[-1] = closing_bulge
    return vertices, bulges, closed
