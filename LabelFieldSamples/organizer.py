#!/usr/bin/env python3
"""
Corn Sample Image Processing Script (minimal, fixed)

What it does
------------
1. Sorts all images by filename sequence.
2. For every TAG file (filename starts with 'tag_'):
      • Try to decode the QR.
      • Record it as a boundary no matter what.
3. For each span between two consecutive TAG files:
      • If the starting TAG decoded, all following regular images in that span
        are assigned to that QR code.
      • If it failed, all those regular images go to <output>/fails/.
4. Copies/renames grouped images into <output>/<qr_code>/.
5. Copies failed TAG images and any unassigned regular images into <output>/fails/.

No CSV/report logic. No artificial limit on number of ear images.
"""

import os
import cv2
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# Optional QR libs
try:
    from pyzbar import pyzbar

    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("Warning: pyzbar not available")

try:
    import zxing

    ZXING_AVAILABLE = True
except ImportError:
    ZXING_AVAILABLE = False
    print("Warning: zxing-cpp not available")


class CornSampleProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fails_dir = self.output_dir / "fails"
        self.fails_dir.mkdir(exist_ok=True)

        self.qr_cache = {}

    # ---------------- filename ordering ----------------
    @staticmethod
    def is_tag(path: Path) -> bool:
        return path.name.lower().startswith("tag_")

    @staticmethod
    def extract_image_sequence_number(image_path: Path) -> int:
        filename = image_path.stem
        clean = filename[4:] if filename.lower().startswith("tag_") else filename

        m = re.match(r"IMG_(\d{8})_(\d{6})", clean, re.IGNORECASE)
        if m:
            return int(m.group(1) + m.group(2))

        m = re.match(r"(\d{8})_(\d{6})", clean)
        if m:
            return int(m.group(1) + m.group(2))

        m = re.match(r"IMG_(\d+)", clean, re.IGNORECASE)
        if m:
            return int(m.group(1))

        nums = re.findall(r"\d+", clean)
        if nums:
            return max(int(n) for n in nums)

        return int(image_path.stat().st_mtime)

    def sort_images_by_sequence(self, image_files):
        print("Sorting images by filename sequence...")
        sequenced = [(self.extract_image_sequence_number(p), p) for p in image_files]
        sequenced.sort(key=lambda x: x[0])
        for i, (seq, p) in enumerate(sequenced, 1):
            tag = " [TAG]" if self.is_tag(p) else ""
            print(f"{i:3d}. {p.name} (seq {seq}){tag}")
        return [p for _, p in sequenced]

    # ---------------- QR decoding ----------------
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        methods = [("original", gray), ("equalized", cv2.equalizeHist(gray))]

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        methods.append(("clahe", clahe_img))

        methods.append(("blurred", cv2.GaussianBlur(gray, (3, 3), 0)))
        methods.append(("bilateral", cv2.bilateralFilter(gray, 9, 75, 75)))

        a1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        methods.append(("adaptive_11_2", a1))

        a2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )
        methods.append(("adaptive_15_4", a2))

        a3 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        methods.append(("adaptive_mean", a3))

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("otsu", otsu))

        for t in [120, 140, 160, 180]:
            _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            methods.append((f"manual_{t}", th))

        a_clahe = cv2.adaptiveThreshold(
            clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        methods.append(("adaptive_clahe", a_clahe))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        methods.append(("closing", cv2.morphologyEx(a1, cv2.MORPH_CLOSE, kernel)))
        methods.append(("opening", cv2.morphologyEx(a1, cv2.MORPH_OPEN, kernel)))

        return methods

    def decode_qr_opencv(self, image):
        detectors = [("opencv_regular", cv2.QRCodeDetector())]
        try:
            detectors.append(("opencv_wechat", cv2.wechat_qrcode_WeChatQRCode()))
        except Exception:
            pass

        for dname, det in detectors:
            for mname, proc in self.preprocess_image(image):
                try:
                    if dname == "opencv_wechat":
                        texts, _ = det.detectAndDecode(proc)
                        if texts:
                            print(f"OpenCV WeChat {mname}: {texts[0]}")
                            return texts[0].strip()
                    else:
                        data, _, _ = det.detectAndDecode(proc)
                        if data:
                            print(f"OpenCV regular {mname}: {data}")
                            return data.strip()
                except Exception:
                    continue
        return None

    def decode_qr_pyzbar(self, image):
        if not PYZBAR_AVAILABLE:
            return None
        for mname, proc in self.preprocess_image(image):
            try:
                codes = pyzbar.decode(proc)
                if codes:
                    data = codes[0].data.decode("utf-8").strip()
                    print(f"Pyzbar {mname}: {data}")
                    return data
            except Exception:
                continue
        return None

    def decode_qr_zxing(self, image_path: Path):
        if not ZXING_AVAILABLE:
            return None
        try:
            reader = zxing.BarCodeReader()
            result = reader.decode(str(image_path))
            if result:
                data = result.text.strip()
                print(f"ZXing: {data}")
                return data
        except Exception:
            pass
        return None

    def decode_qr_code(self, image_path: Path):
        if str(image_path) in self.qr_cache:
            return self.qr_cache[str(image_path)]

        print(f"Attempting to decode QR from: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return None

        decoders = [
            self.decode_qr_opencv,
            self.decode_qr_pyzbar,
            lambda _: self.decode_qr_zxing(image_path),
        ]

        for dec in decoders:
            try:
                res = dec(image)
                if res:
                    self.qr_cache[str(image_path)] = res
                    return res
            except Exception as e:
                print(f"Decoder failed: {e}")
        print(f"Failed to decode QR from: {image_path}")
        return None

    # ---------------- pipeline ----------------
    def process_images(self):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [p for p in self.input_dir.iterdir() if p.suffix.lower() in exts]
        print(f"Found {len(image_files)} image files")

        sorted_imgs = self.sort_images_by_sequence(image_files)

        # Data structures
        qr_groups = defaultdict(list)  # code -> [('qr'/'ear', path, idx), ...]
        qr_tag_paths = defaultdict(list)  # code -> [tag paths]
        unprocessed = []  # [(path, idx)]
        decoded_tag_set = set()
        failed_tag_set = set()
        tag_segments = []  # (idx, code_or_None, tag_path)

        # ---------- FIRST PASS ----------
        print("\n=== FIRST PASS ===")
        for idx, img in enumerate(sorted_imgs):
            print(f"\n#{idx+1}: {img.name}")
            if self.is_tag(img):
                print("  Detected TAG, decoding...")
                code = self.decode_qr_code(img)
                if code:
                    print(f"  SUCCESS: {code}")
                    qr_groups[code].append(("qr", img, idx))
                    qr_tag_paths[code].append(img)
                    decoded_tag_set.add(img)
                    tag_segments.append((idx, code, img))
                else:
                    print("  FAIL decode")
                    failed_tag_set.add(img)
                    tag_segments.append((idx, None, img))
            else:
                unprocessed.append((img, idx))

        # If there are no tags at all, everything goes to fails
        if not tag_segments:
            for img, _ in unprocessed:
                shutil.copy2(img, self.fails_dir / img.name)
                print(f"No tags -> fails: {img.name}")
            return qr_groups

        # ---------- SECOND PASS (segment assignment) ----------
        print("\n=== SECOND PASS ===")
        tag_segments.sort(key=lambda x: x[0])
        boundaries = [pos for (pos, _, _) in tag_segments] + [len(sorted_imgs)]

        for seg_i, (tag_pos, code, tag_path) in enumerate(tag_segments):
            seg_start = tag_pos + 1
            seg_end = boundaries[seg_i + 1]
            print(
                f"\nSegment {seg_i+1}: tag at {tag_pos+1}, "
                f"range {seg_start+1}..{seg_end} "
                f"({'decoded '+code if code else 'FAILED tag'})"
            )

            segment_imgs = [
                (p, i) for (p, i) in unprocessed if seg_start <= i < seg_end
            ]

            if code:  # decoded tag
                cnt = 0
                for p, i in segment_imgs:
                    qr_groups[code].append(("ear", p, i))
                    print(f"  + {p.name} ({i+1})")
                    cnt += 1
                print(f"  Total assigned: {cnt}")
            else:  # failed tag
                for p, _ in segment_imgs:
                    shutil.copy2(p, self.fails_dir / p.name)
                    print(f"  After failed tag -> fails: {p.name}")

        # ---------- move failed tag images ----------
        for path in failed_tag_set:
            shutil.copy2(path, self.fails_dir / path.name)
            print(f"Moved failed tag to fails: {path.name}")

        # ---------- any leftover unprocessed (outside all segments) ----------
        assigned_paths = {img for vals in qr_groups.values() for _, img, _ in vals}
        for p, _ in unprocessed:
            if self.is_tag(p):
                # decoded tags already in groups, failed ones handled above
                continue
            if p not in assigned_paths:
                shutil.copy2(p, self.fails_dir / p.name)
                print(f"Leftover unassigned -> fails: {p.name}")

        # ---------- copy/rename grouped ----------
        for code, items in qr_groups.items():
            self.copy_and_rename_images(code, items, qr_tag_paths[code])

        return qr_groups

    def copy_and_rename_images(self, code, items, tag_paths):
        # tag_paths are all tag images for this code
        ear_imgs = [img for typ, img, _ in items if typ == "ear"]

        code_dir = self.output_dir / code
        code_dir.mkdir(exist_ok=True)

        for i, qimg in enumerate(tag_paths, 1):
            new = f"{code}_tag_{i}{qimg.suffix}"
            shutil.copy2(qimg, code_dir / new)
            print(f"Copied QR: {qimg.name} -> {new}")

        for i, eimg in enumerate(ear_imgs, 1):
            new = f"{code}_ear_{i}{eimg.suffix}"
            shutil.copy2(eimg, code_dir / new)
            print(f"Copied EAR: {eimg.name} -> {new}")

    # ---------------- debug single ----------------
    def debug_single_image(self, image_path, save_debug=True):
        print(f"Debug QR on {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            print("Could not load image.")
            return None
        if save_debug:
            dbg = Path("debug_preprocessing")
            dbg.mkdir(exist_ok=True)
            cv2.imwrite(str(dbg / "00_original.jpg"), image)
            for i, (name, proc) in enumerate(self.preprocess_image(image), 1):
                cv2.imwrite(str(dbg / f"{i:02d}_{name}.jpg"), proc)
        res = self.decode_qr_code(image_path)
        if res:
            print(f"SUCCESS: {res}")
        else:
            print("FAILED")
        return res


def debug_single_qr(image_path):
    p = CornSampleProcessor(".", ".")
    return p.debug_single_image(image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process corn sample images (no CSV/report)"
    )
    parser.add_argument("input_dir", nargs="?", help="Input images dir")
    parser.add_argument("output_dir", nargs="?", help="Output dir")
    parser.add_argument("--debug", type=str, help="Debug single image QR")
    args = parser.parse_args()

    if args.debug:
        if not os.path.exists(args.debug):
            print(f"Debug image not found: {args.debug}")
            return 1
        debug_single_qr(args.debug)
        return 0

    if not all([args.input_dir, args.output_dir]):
        print("Error: input_dir and output_dir are required")
        return 1
    if not os.path.exists(args.input_dir):
        print(f"Input dir not found: {args.input_dir}")
        return 1

    proc = CornSampleProcessor(args.input_dir, args.output_dir)
    print("Starting...")
    print(f"Input : {args.input_dir}")
    print(f"Output: {args.output_dir}")

    proc.process_images()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
