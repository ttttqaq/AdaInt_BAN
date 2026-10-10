import argparse
import binascii
import os
from pathlib import Path
import tempfile
import zlib

from PIL import Image

APP0_MARKER = bytes([0xFF, 0xE0])
APP1_MARKER = bytes([0xFF, 0xE1])
APP2_MARKER = bytes([0xFF, 0xE2])
APP2_MARKER_HEADER = bytes([0x49, 0x43, 0x43, 0x5F, 0x50, 0x52, 0x4F, 0x46, 0x49, 0x4C, 0x45, 0x00, 0x01, 0x01])

DISPLAY_P3_ICC = bytes([
    0x00, 0x00, 0x02, 0x18, 0x61, 0x70, 0x70, 0x6c, 
	0x04, 0x00, 0x00, 0x00, 0x6d, 0x6e, 0x74, 0x72, 
	0x52, 0x47, 0x42, 0x20, 0x58, 0x59, 0x5a, 0x20, 
	0x07, 0xe6, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 
	0x00, 0x00, 0x00, 0x00, 0x61, 0x63, 0x73, 0x70, 
	0x41, 0x50, 0x50, 0x4c, 0x00, 0x00, 0x00, 0x00, 
	0x41, 0x50, 0x50, 0x4c, 0x00, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf6, 0xd6, 
	0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xd3, 0x2d, 
	0x61, 0x70, 0x70, 0x6c, 0xec, 0xfd, 0xa3, 0x8e, 
	0x38, 0x85, 0x47, 0xc3, 0x6d, 0xb4, 0xbd, 0x4f, 
	0x7a, 0xda, 0x18, 0x2f, 0x00, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x00, 0x0a, 0x64, 0x65, 0x73, 0x63, 
	0x00, 0x00, 0x00, 0xfc, 0x00, 0x00, 0x00, 0x30, 
	0x63, 0x70, 0x72, 0x74, 0x00, 0x00, 0x01, 0x2c, 
	0x00, 0x00, 0x00, 0x50, 0x77, 0x74, 0x70, 0x74, 
	0x00, 0x00, 0x01, 0x7c, 0x00, 0x00, 0x00, 0x14, 
	0x72, 0x58, 0x59, 0x5a, 0x00, 0x00, 0x01, 0x90, 
	0x00, 0x00, 0x00, 0x14, 0x67, 0x58, 0x59, 0x5a, 
	0x00, 0x00, 0x01, 0xa4, 0x00, 0x00, 0x00, 0x14, 
	0x62, 0x58, 0x59, 0x5a, 0x00, 0x00, 0x01, 0xb8, 
	0x00, 0x00, 0x00, 0x14, 0x72, 0x54, 0x52, 0x43, 
	0x00, 0x00, 0x01, 0xcc, 0x00, 0x00, 0x00, 0x20, 
	0x63, 0x68, 0x61, 0x64, 0x00, 0x00, 0x01, 0xec, 
	0x00, 0x00, 0x00, 0x2c, 0x62, 0x54, 0x52, 0x43, 
	0x00, 0x00, 0x01, 0xcc, 0x00, 0x00, 0x00, 0x20, 
	0x67, 0x54, 0x52, 0x43, 0x00, 0x00, 0x01, 0xcc, 
	0x00, 0x00, 0x00, 0x20, 0x6d, 0x6c, 0x75, 0x63, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 
	0x00, 0x00, 0x00, 0x0c, 0x65, 0x6e, 0x55, 0x53, 
	0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 
	0x00, 0x44, 0x00, 0x69, 0x00, 0x73, 0x00, 0x70, 
	0x00, 0x6c, 0x00, 0x61, 0x00, 0x79, 0x00, 0x20, 
	0x00, 0x50, 0x00, 0x33, 0x6d, 0x6c, 0x75, 0x63, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 
	0x00, 0x00, 0x00, 0x0c, 0x65, 0x6e, 0x55, 0x53, 
	0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x1c, 
	0x00, 0x43, 0x00, 0x6f, 0x00, 0x70, 0x00, 0x79, 
	0x00, 0x72, 0x00, 0x69, 0x00, 0x67, 0x00, 0x68, 
	0x00, 0x74, 0x00, 0x20, 0x00, 0x41, 0x00, 0x70, 
	0x00, 0x70, 0x00, 0x6c, 0x00, 0x65, 0x00, 0x20, 
	0x00, 0x49, 0x00, 0x6e, 0x00, 0x63, 0x00, 0x2e, 
	0x00, 0x2c, 0x00, 0x20, 0x00, 0x32, 0x00, 0x30, 
	0x00, 0x32, 0x00, 0x32, 0x58, 0x59, 0x5a, 0x20, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf6, 0xd5, 
	0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xd3, 0x2c, 
	0x58, 0x59, 0x5a, 0x20, 0x00, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x83, 0xdf, 0x00, 0x00, 0x3d, 0xbf, 
	0xff, 0xff, 0xff, 0xbb, 0x58, 0x59, 0x5a, 0x20, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a, 0xbf, 
	0x00, 0x00, 0xb1, 0x37, 0x00, 0x00, 0x0a, 0xb9, 
	0x58, 0x59, 0x5a, 0x20, 0x00, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x28, 0x38, 0x00, 0x00, 0x11, 0x0b, 
	0x00, 0x00, 0xc8, 0xb9, 0x70, 0x61, 0x72, 0x61, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 
	0x00, 0x02, 0x66, 0x66, 0x00, 0x00, 0xf2, 0xa7, 
	0x00, 0x00, 0x0d, 0x59, 0x00, 0x00, 0x13, 0xd0, 
	0x00, 0x00, 0x0a, 0x5b, 0x73, 0x66, 0x33, 0x32, 
	0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x42, 
	0x00, 0x00, 0x05, 0xde, 0xff, 0xff, 0xf3, 0x26, 
	0x00, 0x00, 0x07, 0x93, 0x00, 0x00, 0xfd, 0x90, 
	0xff, 0xff, 0xfb, 0xa2, 0xff, 0xff, 0xfd, 0xa3, 
	0x00, 0x00, 0x03, 0xdc, 0x00, 0x00, 0xc0, 0x6e, 
])

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_INPUT_FOLDER = (
    Path(__file__).resolve().parent
    / "work_dirs"
    / "face_indoor_p3_local_color_1"
    / "gen_0"
    / "train_infer"
)
DEFAULT_OUTPUT_FOLDER = DEFAULT_INPUT_FOLDER.with_name("train_infer_p3")


def has_embedded_icc(data: bytes) -> bool:
    """Return whether a JPEG contains an ICC_PROFILE APP2 payload."""
    return b"ICC_PROFILE\x00" in data


def _write_bytes_atomically(path: Path, data: bytes):
    """Write a file beside the source, then replace it atomically."""
    temporary_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(data)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    length = len(payload).to_bytes(4, byteorder="big")
    crc = (binascii.crc32(chunk_type + payload) & 0xFFFFFFFF).to_bytes(4, byteorder="big")
    return length + chunk_type + payload + crc


def _display_p3_png_chunk() -> bytes:
    # PNG iCCP payload: profile name, NUL, compression method, zlib profile data.
    payload = b"Display P3\x00\x00" + zlib.compress(DISPLAY_P3_ICC)
    return _png_chunk(b"iCCP", payload)


def _iter_png_chunks(data: bytes):
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("不是有效的 PNG 文件(缺少 PNG 文件签名)")

    position = len(PNG_SIGNATURE)
    while position + 12 <= len(data):
        length = int.from_bytes(data[position : position + 4], byteorder="big")
        chunk_end = position + 12 + length
        if chunk_end > len(data):
            raise ValueError("不是有效的 PNG 文件(块长度超出文件范围)")
        chunk_type = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + length]
        yield position, chunk_end, chunk_type, payload
        position = chunk_end
        if chunk_type == b"IEND":
            break


def _png_has_display_p3(data: bytes) -> bool:
    for _, _, chunk_type, payload in _iter_png_chunks(data):
        if chunk_type != b"iCCP":
            continue
        try:
            separator = payload.index(0)
            if payload[separator + 1] == 0:
                return zlib.decompress(payload[separator + 2 :]) == DISPLAY_P3_ICC
        except (ValueError, IndexError, zlib.error):
            continue
    return False


def embed_icc_to_png(png_path: str, output_path: str):
    """Add or replace a PNG iCCP chunk without re-encoding image pixels."""
    source_path = Path(png_path)
    destination_path = Path(output_path)
    try:
        data = source_path.read_bytes()
        if _png_has_display_p3(data):
            modified_data = data
        else:
            output = bytearray(PNG_SIGNATURE)
            inserted = False
            found_iend = False
            for start, end, chunk_type, payload in _iter_png_chunks(data):
                if chunk_type == b"iCCP":
                    continue

                output.extend(data[start:end])
                if chunk_type == b"IHDR":
                    output.extend(_display_p3_png_chunk())
                    inserted = True
                if chunk_type == b"IEND":
                    found_iend = True

            if not inserted or not found_iend:
                raise ValueError("不是有效的 PNG 文件(缺少 IHDR 或 IEND)")
            modified_data = bytes(output)

        _write_bytes_atomically(destination_path, modified_data)
        return f"{source_path} — 处理成功, 输出: {destination_path}"
    except Exception as e:
        return f"{source_path} — 处理失败: {e}"

def embed_icc_to_jpg(jpg_path: str, output_path: str):
    source_path = Path(jpg_path)
    destination_path = Path(output_path)
    try:
        data = source_path.read_bytes()

        if has_embedded_icc(data):
            modified_data = data
        else:
            if not data.startswith(b'\xFF\xD8') or not data.rstrip(b'\x00').endswith(b'\xFF\xD9'):
                return f"{source_path} — 处理失败, 不是有效的 JPG 文件(缺少 SOI/EOI),跳过"

            # 完整的 APP2 段内容 = [0xFFE2][Length][ICC_HEADER][ICC_DATA]
            segment_content = APP2_MARKER_HEADER + DISPLAY_P3_ICC
            segment_length = len(segment_content) + 2 #长度占两个字节

            length_field = segment_length.to_bytes(2, byteorder='big')

            app2_segment = APP2_MARKER + length_field + segment_content

            # Keep common JFIF/Exif APP segments before the new ICC APP2 segment.
            insert_pos = 2
            while data[insert_pos : insert_pos + 2] in (APP0_MARKER, APP1_MARKER):
                if insert_pos + 6 > len(data):
                    raise ValueError("APP0/APP1 段长度字段不完整")
                segment_size = int.from_bytes(data[insert_pos + 2 : insert_pos + 4], byteorder="big") + 2
                if segment_size < 4 or insert_pos + segment_size > len(data):
                    raise ValueError("APP0/APP1 段长度无效")
                insert_pos += segment_size

            modified_data = data[:insert_pos] + app2_segment + data[insert_pos:]

        _write_bytes_atomically(destination_path, modified_data)

        return f"{source_path} — 处理成功, 输出: {destination_path}"

    except Exception as e:
        return f"{source_path} — 处理失败: {e}"

def embed_icc_to_bmp_with_convert(bmp_path: str, output_path: str):
    source_path = Path(bmp_path)
    destination_path = Path(output_path).with_suffix(".png")
    temporary_path = None
    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination_path.parent, prefix=f".{destination_path.name}.", suffix=".tmp", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            image.save(temporary_path, format="PNG", icc_profile=DISPLAY_P3_ICC)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_path, destination_path)
        temporary_path = None
        return f"{source_path} — 处理成功, 输出: {destination_path}"
    except Exception as e:
        return f"{source_path} — 处理失败: {e}"
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

def file_process(path: str, output_path: str):
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return embed_icc_to_jpg(path, output_path)
    elif suffix == ".png":
        return embed_icc_to_png(path, output_path)
    elif suffix == ".bmp":
        return embed_icc_to_bmp_with_convert(path, output_path)
    else:
        return f"{path} — 跳过, 非图片格式"   

def process_directory(folder_path: str, output_folder: str):
    """
    遍历输入目录，将处理后的图片写入输出目录并保留相对目录结构。
    """
    source_root = Path(folder_path).resolve()
    destination_root = Path(output_folder).resolve()

    if not source_root.is_dir():
        print(f"错误：指定的文件夹不存在 -> {folder_path}")
        return
    if source_root == destination_root:
        print("错误：输出目录不能与输入目录相同，避免覆盖原始图片")
        return

    destination_root.mkdir(parents=True, exist_ok=True)

    print(f"开始扫描文件夹: {source_root}")
    print(f"处理结果保存到: {destination_root}")
    print("处理条件: 所有 JPG/JPEG/PNG/BMP 文件")
    print("-" * 50)
    
    success_count = 0
    fail_count = 0
    skip_count = 0

    # os.walk 会遍历该文件夹及其所有子文件夹
    for root, dirs, files in os.walk(source_root):
        root_path = Path(root)
        # If the user chooses an output path inside the input tree, do not scan it.
        dirs[:] = [
            directory
            for directory in dirs
            if (root_path / directory).resolve() != destination_root
            and (root_path / directory).resolve() not in destination_root.parents
        ]
        for file in files:
            if Path(file).suffix.lower() not in SUPPORTED_SUFFIXES:
                skip_count += 1
                continue

            file_path = root_path / file
            relative_path = file_path.relative_to(source_root)
            output_path = destination_root / relative_path
            result = file_process(str(file_path), str(output_path))
            print(result)

            if "处理成功" in result:
                success_count += 1
            else:
                fail_count += 1

    print("-" * 50)
    print(f"批量处理完成！共成功处理 {success_count} 张，失败 {fail_count} 张，跳过不符合条件的文件 {skip_count} 个。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="批量为 JPG/JPEG/PNG 图片嵌入 Display P3 ICC 色彩配置。"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(DEFAULT_INPUT_FOLDER),
        help=f"待处理目录，默认: {DEFAULT_INPUT_FOLDER}",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_folder",
        default=str(DEFAULT_OUTPUT_FOLDER),
        help=f"处理结果保存目录，默认: {DEFAULT_OUTPUT_FOLDER}",
    )
    args = parser.parse_args()
    input_folder = os.path.abspath(os.path.expanduser(args.folder))
    output_folder = os.path.abspath(os.path.expanduser(args.output_folder))
    process_directory(input_folder, output_folder)
