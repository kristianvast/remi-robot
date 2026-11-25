"""Convert the staged URDF to a portable USD using Isaac Sim 6.2."""

from pathlib import Path


def main() -> None:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.kit.app

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "isaacsim.asset.importer.urdf", True
        )
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        from pxr import Usd, UsdPhysics

        package = Path(__file__).resolve().parent
        destination = package / "asset"
        destination.mkdir(exist_ok=True)
        result = URDFImporter(
            URDFImporterConfig(
                urdf_path=str(package / "source" / "elegoo_v4.urdf"),
                usd_path=str(destination),
                merge_fixed_joints=True,
                fix_base=False,
                joint_target_type="none",
            )
        ).import_urdf()
        stage = Usd.Stage.Open(str(result))
        stage.SetEditTarget(stage.GetSessionLayer())
        root = stage.GetDefaultPrim()
        variants = root.GetVariantSets().GetVariantSet("Physics")
        if "physx" in variants.GetVariantNames():
            variants.SetVariantSelection("physx")
        joints = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
        collisions = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(joints) != 5 or not collisions:
            raise RuntimeError(f"Invalid converted robot: {joints=}, {collisions=}")
        if not stage.Export(str(destination / "elegoo_v4.usdc")):
            raise RuntimeError("Could not export portable robot USD")
        print(f"CONVERSION_OK root={stage.GetDefaultPrim().GetPath()} {joints=} {collisions=}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
