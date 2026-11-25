"""Split a STEP assembly into one STL per leaf part, in metres, with a JSON index.

Usage: python step_to_parts.py <input.step> <out_dir>

The index (parts.json) lists each part's name, STL file and world-space bounding
box, so links can be built from named groups of parts.
"""

import json
import re
import sys
from pathlib import Path

from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.StlAPI import StlAPI_Writer
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

MM_TO_M = 0.001


def label_name(label: TDF_Label) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return "unnamed"


def walk(shape_tool, label, loc, path, out):
    if XCAFDoc_ShapeTool.IsReference_s(label):
        ref = TDF_Label()
        XCAFDoc_ShapeTool.GetReferredShape_s(label, ref)
        loc = loc * XCAFDoc_ShapeTool.GetLocation_s(label)
        walk(shape_tool, ref, loc, path + [label_name(ref)], out)
        return
    if XCAFDoc_ShapeTool.IsAssembly_s(label):
        children = TDF_LabelSequence()
        XCAFDoc_ShapeTool.GetComponents_s(label, children, False)
        for i in range(1, children.Length() + 1):
            walk(shape_tool, children.Value(i), loc, path, out)
        return
    shape = XCAFDoc_ShapeTool.GetShape_s(label)
    out.append((path or [label_name(label)], shape.Moved(loc)))


def main(step_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    (out / "stl").mkdir(parents=True, exist_ok=True)

    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        sys.exit(f"cannot read {step_path}")
    reader.Transfer(doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    parts = []
    for i in range(1, roots.Length() + 1):
        walk(shape_tool, roots.Value(i), TopLoc_Location(), [], parts)

    index, used = [], {}
    writer = StlAPI_Writer()
    for path, shape in parts:
        BRepMesh_IncrementalMesh(shape, 0.2, False, 0.3, True)
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        if box.IsVoid():
            continue
        lo, hi = box.CornerMin(), box.CornerMax()
        name = path[-1]
        stem = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "part"
        used[stem] = used.get(stem, 0) + 1
        fname = f"{stem}_{used[stem]}.stl"
        writer.Write(shape, str(out / "stl" / fname))
        index.append({
            "name": name,
            "path": path,
            "stl": f"stl/{fname}",
            "bbox_mm": [[lo.X(), lo.Y(), lo.Z()], [hi.X(), hi.Y(), hi.Z()]],
        })

    (out / "parts.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} parts -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
