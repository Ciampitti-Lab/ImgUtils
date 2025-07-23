#!/usr/bin/env python3
"""
HEIC to JPG Converter with Full Metadata Preservation

This script converts HEIC images to JPG format while maintaining all original metadata
including EXIF data, GPS coordinates, timestamps, camera settings, etc.

Requirements:
pip install pillow pillow-heif

Usage:
python heic_converter.py [input_path] [output_path] [--quality QUALITY]

Arguments:
- input_path: Path to HEIC file or directory containing HEIC files
- output_path: Output directory for converted JPG files
- --quality: JPG quality (1-100, default: 95)
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF opener with Pillow
register_heif_opener()


def convert_heic_to_jpg(input_path, output_path, quality=95):
    """
    Convert a single HEIC file to JPG with metadata preservation

    Args:
        input_path (str): Path to input HEIC file
        output_path (str): Path for output JPG file
        quality (int): JPG quality (1-100)

    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Open the HEIC image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (HEIC might be in other color spaces)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Extract all metadata (EXIF, etc.)
            exif_data = img.getexif()

            # Save as JPG with metadata preservation
            img.save(
                output_path,
                "JPEG",
                quality=quality,
                optimize=True,
                exif=exif_data,  # Preserve EXIF metadata
                icc_profile=img.info.get("icc_profile"),  # Preserve color profile
            )

            print(f"✓ Converted: {input_path} → {output_path}")
            return True

    except Exception as e:
        print(f"✗ Error converting {input_path}: {str(e)}")
        return False


def batch_convert(input_dir, output_dir, quality=95):
    """
    Convert all HEIC files in a directory to JPG

    Args:
        input_dir (str): Directory containing HEIC files
        output_dir (str): Output directory for JPG files
        quality (int): JPG quality (1-100)

    Returns:
        tuple: (success_count, total_count)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all HEIC files (case insensitive)
    heic_extensions = [".heic", ".HEIC", ".heif", ".HEIF"]
    heic_files = []

    for ext in heic_extensions:
        heic_files.extend(input_path.glob(f"*{ext}"))

    if not heic_files:
        print(f"No HEIC files found in {input_dir}")
        return 0, 0

    success_count = 0
    total_count = len(heic_files)

    print(f"Found {total_count} HEIC file(s) to convert...")
    print("-" * 50)

    for heic_file in heic_files:
        # Generate output filename
        jpg_filename = heic_file.stem + ".jpg"
        jpg_path = output_path / jpg_filename

        if convert_heic_to_jpg(str(heic_file), str(jpg_path), quality):
            success_count += 1

    return success_count, total_count


def verify_metadata_preservation(original_path, converted_path):
    """
    Verify that metadata was preserved during conversion

    Args:
        original_path (str): Path to original HEIC file
        converted_path (str): Path to converted JPG file
    """
    try:
        with Image.open(original_path) as original:
            original_exif = original.getexif()

        with Image.open(converted_path) as converted:
            converted_exif = converted.getexif()

        print(f"\nMetadata verification for {Path(original_path).name}:")
        print(f"Original EXIF tags: {len(original_exif)}")
        print(f"Converted EXIF tags: {len(converted_exif)}")

        if len(original_exif) > 0:
            preservation_rate = len(converted_exif) / len(original_exif) * 100
            print(f"Metadata preservation: {preservation_rate:.1f}%")

            # Show some key metadata fields if they exist
            key_fields = {
                "DateTime": 306,
                "Camera Make": 271,
                "Camera Model": 272,
                "GPS Latitude": 34853,
                "GPS Longitude": 34855,
            }

            print("\nKey metadata fields:")
            for field_name, tag_id in key_fields.items():
                original_value = original_exif.get(tag_id)
                converted_value = converted_exif.get(tag_id)

                if original_value:
                    status = "✓" if converted_value == original_value else "✗"
                    print(f"  {status} {field_name}: {original_value}")

    except Exception as e:
        print(f"Error verifying metadata: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HEIC images to JPG with metadata preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python heic_converter.py photo.heic output/
  
  # Convert all HEIC files in directory
  python heic_converter.py input_folder/ output_folder/
  
  # Convert with custom quality
  python heic_converter.py input/ output/ --quality 85
        """,
    )

    parser.add_argument("input_path", help="Input HEIC file or directory")
    parser.add_argument("output_path", help="Output directory for JPG files")
    parser.add_argument(
        "--quality", type=int, default=95, help="JPG quality (1-100, default: 95)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify metadata preservation after conversion",
    )

    args = parser.parse_args()

    # Validate quality parameter
    if not 1 <= args.quality <= 100:
        print("Error: Quality must be between 1 and 100")
        sys.exit(1)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)

    # Single file conversion
    if input_path.is_file():
        if input_path.suffix.lower() not in [".heic", ".heif"]:
            print(f"Error: '{input_path}' is not a HEIC file")
            sys.exit(1)

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        jpg_filename = input_path.stem + ".jpg"
        jpg_path = output_path / jpg_filename

        print(f"Converting single file with quality {args.quality}")
        print("-" * 50)

        success = convert_heic_to_jpg(str(input_path), str(jpg_path), args.quality)

        if success and args.verify:
            verify_metadata_preservation(str(input_path), str(jpg_path))

        sys.exit(0 if success else 1)

    # Directory conversion
    elif input_path.is_dir():
        print(f"Converting HEIC files with quality {args.quality}")
        success_count, total_count = batch_convert(
            str(input_path), str(output_path), args.quality
        )

        print("-" * 50)
        print(
            f"Conversion complete: {success_count}/{total_count} files converted successfully"
        )

        if success_count > 0 and args.verify:
            # Verify the first converted file as a sample
            heic_files = list(input_path.glob("*.heic")) + list(
                input_path.glob("*.HEIC")
            )
            if heic_files:
                sample_heic = heic_files[0]
                sample_jpg = output_path / (sample_heic.stem + ".jpg")
                if sample_jpg.exists():
                    verify_metadata_preservation(str(sample_heic), str(sample_jpg))

        sys.exit(0 if success_count == total_count else 1)

    else:
        print(f"Error: '{input_path}' is neither a file nor a directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
