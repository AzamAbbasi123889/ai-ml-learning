"""
main.py
A compact, runnable OCR preprocessing script using pytesseract + OpenCV.
Features (fixed per review):
- Robust confidence parsing
- Token sorting persisted into pages structure
- Normalized detected_fields objects: consistent bbox + level
- bbox helpers documented (expand_bbox returns None for empty input)
- Optional PDF support (requires poppler + pdf2image)

Usage example:
python main.py --image tests/sample_receipt.jpg --out result.json --visualize

Note: This script requires the Tesseract binary installed on your system.
On Debian/Ubuntu: sudo apt-get install -y tesseract-ocr
If tesseract isn't in PATH, set pytesseract.pytesseract.tesseract_cmd to the full binary path.
"""

import argparse
import json
import re
from typing import List, Optional

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Optional PDF support
try:
    from pdf2image import convert_from_path
    _PDF2IMAGE_AVAILABLE = True
except Exception:
    _PDF2IMAGE_AVAILABLE = False

# Simple illustrative regexes — documented as intentionally limited below
DATE_RE = re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b")
CURRENCY_RE = re.compile(r"\b(?:[$€£]|USD|EUR|GBP)?\s*\d{1,3}(?:[\,\s]\d{3})*(?:\.\d{1,2})?\b", re.IGNORECASE)


def safe_parse_conf(raw_conf) -> float:
    """Convert pytesseract confidence string to float safely.

    pytesseract returns conf values in Output.DICT as strings. Depending on Tesseract/pytesseract versions,
    values may be: "85", "85.0", "-1", "-1.0", or an empty string. Return -1.0 for unparsable values.
    """
    try:
        return float(raw_conf)
    except (ValueError, TypeError):
        return -1.0


def bbox_from_ltrwh(left: int, top: int, width: int, height: int) -> List[int]:
    return [int(left), int(top), int(left + width), int(top + height)]


def expand_bbox(bbox: Optional[List[int]], pad: int = 0) -> Optional[List[int]]:
    """Return a new bbox expanded by `pad` pixels on all sides.

    If bbox is None or empty, return None. This function explicitly documents that None may be returned
    and callers should omit bbox fields when None.
    """
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return [x1 - pad, y1 - pad, x2 + pad, y2 + pad]


def run_ocr_on_image(img: np.ndarray, min_conf: float = 30.0):
    """Run pytesseract OCR on an OpenCV image and build a pages dict structure.

    Returns a result dict with normalized data and consistent ordering.
    Token order in pages[...]['tokens'] is persisted as visual order (left->right within lines).
    """
    h, w = img.shape[:2]
    data = pytesseract.image_to_data(img, output_type=Output.DICT)

    n = len(data.get('level', []))
    pages = {1: {'width': w, 'height': h, 'tokens': [], 'lines': [], 'paragraphs': []}}

    # iterate words and collect tokens
    for i in range(n):
        text = data.get('text', [None] * n)[i]
        if text is None:
            continue
        raw_conf = data.get('conf', ["-1"] * n)[i]
        conf = safe_parse_conf(raw_conf)
        left = data.get('left', [0] * n)[i]
        top = data.get('top', [0] * n)[i]
        width = data.get('width', [0] * n)[i]
        height = data.get('height', [0] * n)[i]
        block_num = data.get('block_num', [0] * n)[i]
        par_num = data.get('par_num', [0] * n)[i]
        line_num = data.get('line_num', [0] * n)[i]
        word_num = data.get('word_num', [0] * n)[i]

        token = {
            'text': text,
            'conf': conf,
            'bbox': bbox_from_ltrwh(left, top, width, height),
            'block_num': int(block_num),
            'par_num': int(par_num),
            'line_num': int(line_num),
            'word_num': int(word_num),
        }
        pages[1]['tokens'].append(token)

    # group tokens into lines and paragraphs using block/par/line ids
    # Build a mapping: (block,par,line) -> list of tokens
    from collections import defaultdict

    line_map = defaultdict(list)
    par_map = defaultdict(list)

    for t in pages[1]['tokens']:
        key_line = (t['block_num'], t['par_num'], t['line_num'])
        key_par = (t['block_num'], t['par_num'])
        line_map[key_line].append(t)
        par_map[key_par].append(t)

    # Create line objects with tokens sorted left-to-right
    lines = []
    for key, toks in line_map.items():
        # sort tokens visually: by left (bbox[0]) then by top as tie-breaker
        toks_sorted = sorted(toks, key=lambda tt: (tt['bbox'][0], tt['bbox'][1]))
        # persist sorted order back into the pages tokens list if desired: replace matching tokens
        # For simplicity, we'll update the tokens list to reflect visual ordering within the same line
        # This makes pages[1]['tokens'] contain tokens in an order that reflects visual grouping by lines
        # (full sorting across the page can be implemented if needed)
        # Replace original tokens in pages[1]['tokens'] by matching identity: here we just update their relative order
        # Find indices of these tokens and reorder them in the global list
        # Build a map from (block,par,line,word) to token
        lines.append({
            'block_num': key[0],
            'par_num': key[1],
            'line_num': key[2],
            'tokens': toks_sorted,
            'bbox': None,
        })

    # compute canonical bbox for lines
    for line in lines:
        xs = [t['bbox'][0] for t in line['tokens']] + [t['bbox'][2] for t in line['tokens']]
        ys = [t['bbox'][1] for t in line['tokens']] + [t['bbox'][3] for t in line['tokens']]
        line['bbox'] = [min(xs), min(ys), max(xs), max(ys)]

    # Persist the token sorting into pages[1]['tokens'] by reconstructing tokens grouped by line (top->bottom)
    lines_sorted_by_top = sorted(lines, key=lambda l: (l['bbox'][1], l['bbox'][0]))
    new_tokens_order = []
    for l in lines_sorted_by_top:
        new_tokens_order.extend(l['tokens'])
    pages[1]['tokens'] = new_tokens_order

    # Build paragraphs
    paragraphs = []
    for key, toks in par_map.items():
        toks_sorted = sorted(toks, key=lambda tt: (tt['bbox'][1], tt['bbox'][0]))
        xs = [t['bbox'][0] for t in toks_sorted] + [t['bbox'][2] for t in toks_sorted]
        ys = [t['bbox'][1] for t in toks_sorted] + [t['bbox'][3] for t in toks_sorted]
        paragraphs.append({
            'block_num': key[0],
            'par_num': key[1],
            'tokens': toks_sorted,
            'bbox': [min(xs), min(ys), max(xs), max(ys)] if toks_sorted else None,
        })

    pages[1]['lines'] = lines_sorted_by_top
    pages[1]['paragraphs'] = paragraphs

    # Detected fields: normalized schema
    detected_fields = []
    for t in pages[1]['tokens']:
        txt = t['text']
        if DATE_RE.search(txt):
            detected_fields.append({
                'type': 'date',
                'text': txt,
                'page': 1,
                'level': 'token',
                'bbox': t['bbox'],
                'confidence': t['conf'],
            })
        if CURRENCY_RE.search(txt):
            detected_fields.append({
                'type': 'currency',
                'text': txt,
                'page': 1,
                'level': 'token',
                'bbox': t['bbox'],
                'confidence': t['conf'],
            })

    # line-level simple detection (e.g., a whole line looks like a date or amount)
    for line in pages[1]['lines']:
        line_text = ' '.join([tt['text'] for tt in line['tokens']])
        if DATE_RE.search(line_text):
            detected_fields.append({
                'type': 'date',
                'text': line_text,
                'page': 1,
                'level': 'line',
                'bbox': line['bbox'],
                'confidence': float(np.mean([tt['conf'] for tt in line['tokens']]) if line['tokens'] else -1.0),
            })

    result = {
        'pages': pages,
        'detected_fields': detected_fields,
    }

    return result


def visualize_result(img: np.ndarray, result: dict, outpath: str):
    vis = img.copy()
    for field in result.get('detected_fields', []):
        bbox = field.get('bbox')
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(vis, f"{field['type']}:{field['text']}", (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)
    cv2.imwrite(outpath, vis)


def image_from_path(path: str) -> List[np.ndarray]:
    """Load a single image or (if path ends with .pdf and pdf2image available) convert pages.

    Returns a list of OpenCV images (BGR).
    """
    if path.lower().endswith('.pdf'):
        if not _PDF2IMAGE_AVAILABLE:
            raise RuntimeError('pdf2image not available. Install pdf2image and ensure poppler is installed.')
        pil_pages = convert_from_path(path)
        imgs = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pil_pages]
        return imgs
    else:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f'Image not found or cannot be read: {path}')
        return [img]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image (jpg/png) or PDF (optional)')
    parser.add_argument('--out', default='result.json', help='Output JSON path')
    parser.add_argument('--visualize', action='store_true', help='Write a visualization image with detections')
    parser.add_argument('--tesseract-cmd', default=None, help='Full path to tesseract binary if not in PATH')
    args = parser.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    images = image_from_path(args.image)
    results = {'pages': {}, 'detected_fields': []}
    for i, img in enumerate(images, start=1):
        r = run_ocr_on_image(img)
        # pages are keyed by 1 in run_ocr_on_image; adjust for multi-page
        results['pages'][str(i)] = r['pages'][1]
        # add detected fields with page index corrected
        for f in r['detected_fields']:
            f_copy = f.copy()
            f_copy['page'] = i
            results['detected_fields'].append(f_copy)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if args.visualize:
        vis_out = args.out + '.png'
        visualize_result(images[0], results, vis_out)
        print('Wrote visualization to', vis_out)

    print('Wrote JSON to', args.out)


if __name__ == '__main__':
    main()
