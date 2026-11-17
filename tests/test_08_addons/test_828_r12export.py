#  Copyright (c) 2026, Manfred Moitzi
#  License: MIT License
"""Tests for the DXF R12 export add-on (ezdxf.addons.r12export)."""
import pytest

import ezdxf
from ezdxf import path
from ezdxf.addons import r12export
from ezdxf.lldxf.tagger import internal_tag_compiler
from ezdxf.math import Vec3


@pytest.fixture
def doc():
    return ezdxf.new("R2010")


def add_solid_mpolygon(layout, offset=(0, 0), extrusion=(0, 0, 1)):
    mp = layout.add_mpolygon(color=3)
    mp.set_solid_fill()
    mp.paths.add_polyline_path(
        [(0, 0), (10, 0), (10, 10), (0, 10)], is_closed=True
    )
    mp.dxf.offset_vector = offset
    mp.dxf.extrusion = extrusion
    return mp


def add_pattern_mpolygon(layout, offset=(0, 0), extrusion=(0, 0, 1)):
    mp = layout.add_mpolygon(color=3)
    mp.set_pattern_fill("ANSI31")
    mp.paths.add_polyline_path(
        [(0, 0), (10, 0), (10, 10), (0, 10)], is_closed=True
    )
    mp.dxf.offset_vector = offset
    mp.dxf.extrusion = extrusion
    return mp


def get_anonymous_block(doc, insert):
    return doc.blocks.get(insert.dxf.name)


class TestPaperspaceExport:
    """Converted entities of the paperspace must keep the paperspace flag
    (group code 67), otherwise they end up in the modelspace.
    """

    @pytest.fixture
    def psp_doc(self, doc):
        psp = doc.paperspace()
        psp.add_lwpolyline([(0, 0), (10, 0), (10, 10)])
        psp.add_spline([(0, 0), (5, 5), (10, 0)])
        psp.add_ellipse((5, 5), major_axis=(2, 0), ratio=0.5)
        psp.add_mtext("hello", dxfattribs={"char_height": 1})
        hatch = psp.add_hatch()
        hatch.paths.add_polyline_path([(0, 0), (1, 0), (1, 1)], is_closed=True)
        return doc

    def test_converted_entities_stay_in_paperspace(self, psp_doc):
        result = r12export.convert(psp_doc)
        assert len(result.modelspace()) == 0
        types = sorted(e.dxftype() for e in result.paperspace())
        # LWPOLYLINE, SPLINE and ELLIPSE are converted to POLYLINE entities,
        # MTEXT is exploded to TEXT and HATCH is exported as INSERT of an
        # anonymous block
        assert types == ["INSERT", "POLYLINE", "POLYLINE", "POLYLINE", "TEXT"]

    def test_converted_entities_have_paperspace_flag_set(self, psp_doc):
        exporter = r12export.R12Exporter(psp_doc)
        tags = list(internal_tag_compiler(exporter.to_string()))
        # extract the ENTITIES section
        start = next(
            i for i, t in enumerate(tags[:-1]) if t == (0, "SECTION") and tags[i + 1] == (2, "ENTITIES")
        )
        end = next(i for i, t in enumerate(tags) if i > start and t == (0, "ENDSEC"))
        # every exported entity and all its sub-entities (VERTEX) following a
        # (0, entity-type) tag up to the next (0, ...) tag need the paperspace
        # flag (67, 1)
        current = None
        flagged = []
        for tag in tags[start:end]:
            if tag[0] == 0:
                current = tag[1]
            elif tag == (67, 1):
                flagged.append(current)
        for entity_type in ("POLYLINE", "VERTEX", "TEXT", "INSERT"):
            assert entity_type in flagged, f"{entity_type} without paperspace flag"

    def test_modelspace_entities_have_no_paperspace_flag(self, doc):
        doc.modelspace().add_lwpolyline([(0, 0), (10, 0), (10, 10)])
        result = r12export.convert(doc)
        assert len(result.paperspace()) == 0
        assert len(result.modelspace()) == 1


class TestMultiplePaperspaceLayouts:
    """DXF R12 supports only a single paperspace: all paperspace layouts have
    to be merged into this one paperspace.
    """

    @pytest.fixture
    def multi_layout_doc(self, doc):
        doc.paperspace().add_circle((0, 0), 2)
        sheet2 = doc.layouts.new("Sheet2")
        sheet2.add_line((0, 0), (10, 10))
        sheet3 = doc.layouts.new("Sheet3")
        block = doc.blocks.new("MYBLK")
        block.add_circle((0, 0), 1)
        sheet3.add_blockref("MYBLK", (5, 5))
        return doc

    def test_entities_of_all_layouts_are_exported(self, multi_layout_doc):
        result = r12export.convert(multi_layout_doc)
        types = set(e.dxftype() for e in result.paperspace())
        assert types == {"CIRCLE", "LINE", "INSERT"}

    def test_block_references_of_all_layouts_are_exported(self, multi_layout_doc):
        result = r12export.convert(multi_layout_doc)
        insert = result.paperspace().query("INSERT").first
        assert insert is not None
        assert insert.dxf.name == "MYBLK"
        assert "MYBLK" in result.blocks

    def test_result_has_a_single_paperspace(self, multi_layout_doc):
        result = r12export.convert(multi_layout_doc)
        layouts = [layout for layout in result.layouts if layout.is_any_paperspace]
        assert len(layouts) == 1


class TestMPolygonOffsetVector:
    """The MPOLYGON offset vector (group code 11/21) is located in the OCS and
    has to be applied to the boundary paths, the solid fill and the pattern
    fill.
    """

    def test_export_does_not_raise_dxf_attribute_error(self, doc):
        add_solid_mpolygon(doc.modelspace(), offset=(2, 3))
        # raised DXFAttributeError: Invalid DXF attribute "offset" before the fix
        result = r12export.convert(doc)
        assert len(result.modelspace()) == 1

    def test_solid_fill_is_shifted_by_offset(self, doc):
        add_solid_mpolygon(doc.modelspace(), offset=(2, 3))
        result = r12export.convert(doc)
        insert = result.modelspace().query("INSERT").first
        block = get_anonymous_block(result, insert)
        solids = list(block.query("SOLID"))
        assert len(solids) > 0
        xs = [v.x for s in solids for v in s.vertices()]
        ys = [v.y for s in solids for v in s.vertices()]
        assert min(xs) == pytest.approx(2.0)
        assert max(xs) == pytest.approx(12.0)
        assert min(ys) == pytest.approx(3.0)
        assert max(ys) == pytest.approx(13.0)

    def test_boundary_path_is_shifted_by_offset(self, doc):
        add_solid_mpolygon(doc.modelspace(), offset=(2, 3))
        result = r12export.convert(doc)
        insert = result.modelspace().query("INSERT").first
        block = get_anonymous_block(result, insert)
        polyline = block.query("POLYLINE").first
        assert polyline is not None
        xs = [v.dxf.location.x for v in polyline.vertices]
        ys = [v.dxf.location.y for v in polyline.vertices]
        assert min(xs) == pytest.approx(2.0)
        assert max(xs) == pytest.approx(12.0)
        assert min(ys) == pytest.approx(3.0)
        assert max(ys) == pytest.approx(13.0)

    def test_pattern_fill_is_shifted_by_offset(self, doc):
        add_pattern_mpolygon(doc.modelspace(), offset=(5, 5))
        result = r12export.convert(doc)
        insert = result.modelspace().query("INSERT").first
        block = get_anonymous_block(result, insert)
        lines = list(block.query("LINE"))
        assert len(lines) > 0
        xs = [v.x for line in lines for v in (line.dxf.start, line.dxf.end)]
        ys = [v.y for line in lines for v in (line.dxf.start, line.dxf.end)]
        assert min(xs) == pytest.approx(5.0)
        assert max(xs) == pytest.approx(15.0)
        assert min(ys) == pytest.approx(5.0)
        assert max(ys) == pytest.approx(15.0)

    def test_non_z_extrusion_boundary_and_fill_are_consistent(self, doc):
        # the OCS of the extrusion vector (0, 0, -1) maps the OCS x-axis to
        # the negated WCS x-axis: OCS (x, y) -> WCS (-x, y)
        add_pattern_mpolygon(doc.modelspace(), offset=(1, 1), extrusion=(0, 0, -1))
        result = r12export.convert(doc)
        insert = result.modelspace().query("INSERT").first
        block = get_anonymous_block(result, insert)

        polyline = block.query("POLYLINE").first
        assert polyline is not None
        assert Vec3(polyline.dxf.extrusion).isclose((0, 0, -1))
        # boundary and pattern fill have to coincide in the WCS:
        from ezdxf.math import OCS

        ocs = OCS((0, 0, -1))
        wcs_vertices = [
            ocs.to_wcs(Vec3(v.dxf.location)) for v in polyline.vertices
        ]
        xs = [v.x for v in wcs_vertices]
        ys = [v.y for v in wcs_vertices]
        assert min(xs) == pytest.approx(-11.0)
        assert max(xs) == pytest.approx(-1.0)
        assert min(ys) == pytest.approx(1.0)
        assert max(ys) == pytest.approx(11.0)

        lines = list(block.query("LINE"))
        assert len(lines) > 0
        xs = [v.x for line in lines for v in (line.dxf.start, line.dxf.end)]
        ys = [v.y for line in lines for v in (line.dxf.start, line.dxf.end)]
        assert min(xs) == pytest.approx(-11.0)
        assert max(xs) == pytest.approx(-1.0)
        assert min(ys) == pytest.approx(1.0)
        assert max(ys) == pytest.approx(11.0)


class TestPolygonTools:
    """Low level tools used by the R12 export and user scripts."""

    def test_triangulate_applies_mpolygon_offset_vector(self, doc):
        mp = add_solid_mpolygon(doc.modelspace(), offset=(2, 3))
        triangles = list(mp.triangulate(max_sagitta=0.01))
        assert len(triangles) > 0
        xs = [v.x for t in triangles for v in t]
        ys = [v.y for t in triangles for v in t]
        assert min(xs) == pytest.approx(2.0)
        assert max(xs) == pytest.approx(12.0)
        assert min(ys) == pytest.approx(3.0)
        assert max(ys) == pytest.approx(13.0)

    def test_triangulate_hatch_is_not_affected(self, doc):
        hatch = doc.modelspace().add_hatch()
        hatch.paths.add_polyline_path(
            [(0, 0), (10, 0), (10, 10), (0, 10)], is_closed=True
        )
        triangles = list(hatch.triangulate(max_sagitta=0.01))
        xs = [v.x for t in triangles for v in t]
        assert min(xs) == pytest.approx(0.0)
        assert max(xs) == pytest.approx(10.0)

    def test_from_hatch_ocs_applies_offset(self, doc):
        mp = add_solid_mpolygon(doc.modelspace(), offset=(2, 3))
        paths = list(path.from_hatch_ocs(mp, offset=Vec3(mp.dxf.offset_vector)))
        assert len(paths) > 0
        vertices = [v for p in paths for v in p.flattening(0.01)]
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]
        assert min(xs) == pytest.approx(2.0)
        assert max(xs) == pytest.approx(12.0)
        assert min(ys) == pytest.approx(3.0)
        assert max(ys) == pytest.approx(13.0)

    def test_render_pattern_lines_applies_mpolygon_offset(self, doc):
        mp = add_pattern_mpolygon(doc.modelspace(), offset=(5, 5))
        lines = list(mp.render_pattern_lines())
        assert len(lines) > 0
        xs = [v.x for line in lines for v in line]
        ys = [v.y for line in lines for v in line]
        assert min(xs) == pytest.approx(5.0)
        assert max(xs) == pytest.approx(15.0)
        assert min(ys) == pytest.approx(5.0)
        assert max(ys) == pytest.approx(15.0)

    def test_render_pattern_lines_hatch_is_not_affected(self, doc):
        hatch = doc.modelspace().add_hatch()
        hatch.set_pattern_fill("ANSI31")
        hatch.paths.add_polyline_path(
            [(0, 0), (10, 0), (10, 10), (0, 10)], is_closed=True
        )
        lines = list(hatch.render_pattern_lines())
        assert len(lines) > 0
        xs = [v.x for line in lines for v in line]
        assert min(xs) == pytest.approx(0.0)
        assert max(xs) == pytest.approx(10.0)


if __name__ == "__main__":
    pytest.main([__file__])
