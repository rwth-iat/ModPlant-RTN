from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreparedSchemaBundle:
    id: str
    directory: Path
    source_checksum: str


def directory_checksum(directory: str | Path) -> str:
    digest = hashlib.sha256()
    root = Path(directory)
    for path in sorted(root.glob("*.xsd")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_schema_bundle(
    bundle_id: str,
    mesa_schema_directory: str | Path,
    destination: str | Path,
) -> PreparedSchemaBundle:
    """Materialize exactly one schema bundle in an isolated directory.

    The MESA source is never edited. The Module bundle is made by copying the
    complete package and replacing selected extension group definitions in that
    copy, avoiding duplicate definitions in the shared extension namespace.
    """
    if bundle_id not in {"mesa-v7.01", "modplant-profile-v2"}:
        raise ValueError(f"Unknown schema bundle: {bundle_id}")
    source = Path(mesa_schema_directory).resolve()
    target = Path(destination).resolve()
    if not (source / "B2MML-AllExtensions.xsd").exists():
        raise FileNotFoundError("The directory is not a complete MESA BatchML schema bundle")
    checksum = directory_checksum(source)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    if bundle_id == "modplant-profile-v2":
        _install_module_extension_groups(target)
    manifest_source = Path(__file__).resolve().parents[1] / "schema_bundles" / bundle_id / "manifest.json"
    shutil.copy2(manifest_source, target / "module-bundle-manifest.json")
    return PreparedSchemaBundle(bundle_id, target, checksum)


def _install_module_extension_groups(directory: Path) -> None:
    all_extensions = directory / "B2MML-AllExtensions.xsd"
    text = all_extensions.read_text(encoding="utf-8")
    declarations = """
  <xsd:element name="ModuleEdgeSemantics" type="xsd:string"/>
  <xsd:element name="ModuleGatewaySemantics" type="xsd:string"/>
  <xsd:element name="ModuleConditionAST" type="xsd:string"/>
"""
    text = text.replace("</xsd:schema>", declarations + "\n</xsd:schema>")
    all_extensions.write_text(text, encoding="utf-8")

    patches = {
        "BatchML-GeneralRecipeExtensions.xsd": {
            "DirectedLink": "ModuleEdgeSemantics",
            "ProcedureChartElement": "ModuleGatewaySemantics",
        },
        "BatchML-BatchInformationExtensions.xsd": {
            "Link": "ModuleGatewaySemantics",
            "Transition": "ModuleConditionAST",
        },
    }
    for filename, groups in patches.items():
        path = directory / filename
        content = path.read_text(encoding="utf-8")
        for group_name, element_name in groups.items():
            pattern = rf'(<xsd:group\s+name\s*=\s*"{re.escape(group_name)}"\s*>\s*<xsd:sequence>)'
            replacement = rf'\1\n          <xsd:element ref="{element_name}" minOccurs="0" maxOccurs="1"/>'
            content, count = re.subn(pattern, replacement, content, count=1)
            if count != 1:
                raise ValueError(f"Could not replace extension group {group_name} in {filename}")
        path.write_text(content, encoding="utf-8")

