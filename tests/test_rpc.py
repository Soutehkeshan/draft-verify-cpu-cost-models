import tarfile

import tvm
from tvm import te

from dvcm.rpc import export_rpc_archive


def test_export_rpc_archive_omits_macos_resource_fork_members(tmp_path):
    x = te.placeholder((8,), dtype="float32", name="x")
    y = te.compute((8,), lambda i: x[i] + 1.0, name="y")
    prim = te.create_prim_func([x, y]).with_attr("global_symbol", "add_one")
    library = tvm.compile(tvm.IRModule.from_expr(prim), target="llvm")
    archive_path = tmp_path / "module.tar"

    export_rpc_archive(library, archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getnames()
    assert members
    assert all(not member.rsplit("/", 1)[-1].startswith("._") for member in members)
