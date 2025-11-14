from markitdown import MarkItDown
from pathlib import Path
from typing import Union

def convert_and_save(source_path: Union[str, Path]) -> Path:
    """
    Converts a file to Markdown and saves the result.
    
    Args:
        source_path: Path to the source file
        
    Returns:
        Path: Path to the created Markdown file
        
    Raises:
        FileNotFoundError: If the source file does not exist
        PermissionError: If write permissions are lacking
    """
    md = MarkItDown()
    source = Path(source_path).expanduser().resolve()
    
    # Validations
    if not source.exists():
        raise FileNotFoundError(f"File {source} does not exist")
    
    if not source.is_file():
        raise ValueError(f"{source} is not a file")
    
    # Conversion
    result = md.convert(str(source))
    
    # Save
    output_path = source.with_suffix('.md')
    output_path.write_text(result.text_content, encoding='utf-8')
    
    return output_path


if __name__ == "__main__":
    home = Path.home()
    downloads = home / "Downloads"
    source = downloads / "StructuredRagPyBay25.pdf"

    output_file = convert_and_save(source)
    print(f"Markdown file created: {output_file}")