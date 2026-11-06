"""Safely inspect + extract the downloaded historical image archive.

Images only: refuses entries that are not images or that try to escape the
target directory. Run: py -3.12 -I mlr-ex08-tools/unpack_history.py <zip> <dest>
"""
import os
import sys
import zipfile

IMG = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
TEXT = (".csv", ".txt", ".md")          # metadata we may read, never execute

zip_path, dest = sys.argv[1], sys.argv[2]
archive = zipfile.ZipFile(zip_path)
names = archive.namelist()
print("entries:", len(names))

bad = []
for name in names:
    if name.endswith("/"):
        continue
    if not name.lower().endswith(IMG + TEXT):
        bad.append(("not-an-image", name))
    if ".." in name or name.startswith(("/", "\\")) or ":" in name:
        bad.append(("unsafe-path", name))
if bad:
    print("REFUSING, suspicious entries:", bad[:10])
    sys.exit(1)

archive.extractall(dest)
print("extracted to", dest)
for root, dirs, files in os.walk(dest):
    if files:
        print(f"{os.path.relpath(root, dest)}: {len(files)} files")
