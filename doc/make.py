#!/usr/bin/env python3
import os
import pathlib as pl
import shutil
from collections import defaultdict
from typing import Any

import lazydocs


def build_toc(files: list[str]) -> dict[str, Any]:
    toc: dict[str, Any] = defaultdict(lambda: defaultdict(dict))

    for file in files:
        parts = file[:-3].rsplit(".", maxsplit=file.count(".") - 1)
        current_level = toc

        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = defaultdict(dict)
            current_level = current_level[part]
        current_level[file] = file

    return toc


def stringify_toc(toc: dict[str, Any], level: int = 0) -> str:
    toc_string = ""
    indent = "  " * level

    for key, value in toc.items():
        if isinstance(value, dict):
            toc_string += stringify_toc(value, level + 1)
        else:
            toc_string += f"{indent}- [{key.split('.')[-2]}](./api/{value})\n"

    return toc_string


def main():
    here = pl.Path(__file__).parent
    out = here / "src" / "api"
    summary_tpl = here / "SUMMARY.tpl"
    summary_md = here / "src" / "SUMMARY.md"

    # Generate docs
    if out.exists():
        shutil.rmtree(out)
    lazydocs.generate_docs(
        paths=[str(here.parent / "src")],
        output_path=str(out),
        src_base_url=(   # UPDATE WITH V0.1.0 TO GRAB TAG
            "https://github.com/childmindresearch/niftyone/tree/main"
        ),
        watermark=False,
        ignored_modules=["niclips", "niftyone.typing"],
    )

    # Grab and organize files
    toc = build_toc(
        sorted(os.listdir(out), key=lambda f: (f[:-3].split("."), len(f.split("."))))
    )
    toc = stringify_toc(toc)

    # Update summary via template - Will automatically create the SUMMARY.md file
    with open(summary_tpl, "r") as summary_file, open(summary_md, "w") as out_fpath:
        content = summary_file.read()
        out_fpath.write(content.replace("{{api_toc}}", toc))


if __name__ == "__main__":
    main()
