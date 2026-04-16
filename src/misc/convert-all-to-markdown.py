# /// script
# dependencies = ["markitdown[all]>=0.1.3"]
# ///

from markitdown import MarkItDown
from pathlib import Path
from typing import Union, Iterable

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}


def convert_and_save(source_path: Union[str, Path], md: MarkItDown = None) -> Path:
    """
    Converts a file to Markdown and saves the result next to the source file.
    """
    if md is None:
        md = MarkItDown()
    source = Path(source_path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"File {source} does not exist")
    if not source.is_file():
        raise ValueError(f"{source} is not a file")

    result = md.convert(str(source))

    output_path = source.with_suffix(".md")
    output_path.write_text(result.text_content, encoding="utf-8")

    return output_path


def convert_directory(
    directory: Union[str, Path],
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    recursive: bool = True,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Converts all files with the given extensions in a directory (and subdirectories)
    to Markdown.

    Args:
        directory: Root directory to scan.
        extensions: Iterable of file extensions to include (with leading dot).
        recursive: Whether to traverse subdirectories.
        skip_existing: If True, skip files whose .md counterpart already exists.

    Returns:
        list[Path]: Paths of the Markdown files that were created.
    """
    directory = Path(directory).expanduser().resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    # Normalise extensions to lowercase with leading dot
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    iterator = directory.rglob("*") if recursive else directory.glob("*")
    files = sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in exts
    )

    if not files:
        print(f"No files with extensions {sorted(exts)} found in {directory}")
        return []

    print(f"Found {len(files)} file(s) to convert\n")

    md = MarkItDown()
    converted: list[Path] = []

    for i, src in enumerate(files, 1):
        rel = src.relative_to(directory)
        try:
            output_path = src.with_suffix(".md")
            if skip_existing and output_path.exists():
                print(f"[{i}/{len(files)}] Skipped (already exists): {rel}")
                continue

            print(f"[{i}/{len(files)}] Converting: {rel}")
            output = convert_and_save(src, md=md)
            converted.append(output)
            print(f"    ✓ Created: {output.relative_to(directory)}")
        except Exception as e:
            print(f"    ✗ Error on {rel}: {e}")

    return converted


if __name__ == "__main__":
    directory = Path("/Users/alain/Downloads/Groupe")

    converted_files = convert_directory(directory)
    print(f"\n{len(converted_files)} file(s) successfully converted to Markdown.")

