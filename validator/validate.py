import os
import sys
import json
import argparse
import re
import numpy as np
from PIL import Image

# Import hàm đọc PNG chuẩn của bài lab
try:
    from scripts.pngutil import read_png_id
except ImportError:
    def read_png_id(png_path):
        img = Image.open(png_path).convert('RGB')
        arr = np.array(img, dtype=np.uint32)
        return arr[:, :, 0] + arr[:, :, 1] * 256 + arr[:, :, 2] * (256 ** 2)

COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

CLASS_SYNONYMS = {
    'couch': 'sofa',
    'sofa': 'couch',
    'tvmonitor': 'tv',
    'tv monitor': 'tv',
    'pottedplant': 'potted plant',
    'plant': 'potted plant',
    'diningtable': 'dining table',
    'table': 'dining table',
    'cellphone': 'cell phone',
    'mobile phone': 'cell phone',
    'phone': 'cell phone',
    'motorbike': 'motorcycle',
    'aeroplane': 'airplane'
}

def normalize_name(name):
    if not name:
        return ""
    # Xóa số tiền tố ở đầu chuỗi (ví dụ: "0 person" -> "person")
    name = re.sub(r'^\d+\s*', '', str(name))
    clean = name.lower().strip().replace('_', ' ').replace('-', ' ')
    clean = " ".join(clean.split())
    return CLASS_SYNONYMS.get(clean, clean)

def compute_box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def find_file_or_dir(base_dir, candidates):
    for c in candidates:
        p = os.path.join(base_dir, c)
        if os.path.exists(p):
            return p
    return None

def load_class_names(delivery_dir):
    classes_file = find_file_or_dir(delivery_dir, [
        'classes.txt', 'labels/classes.txt', 'det_labels/classes.txt'
    ])
    if classes_file and os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            raw_lines = [l.strip() for l in f if l.strip()]
            parsed_names = []
            for line in raw_lines:
                # Bóc tách bỏ số thứ tự ở đầu dòng "0 person" -> "person"
                clean_line = re.sub(r'^\d+\s+', '', line).strip()
                parsed_names.append(clean_line)
            if parsed_names:
                return parsed_names
    return COCO_80_CLASSES

def parse_yolo_txt(txt_path, img_w, img_h, class_names):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])

            x1 = int(max(0, (x_c - w / 2) * img_w))
            y1 = int(max(0, (y_c - h / 2) * img_h))
            x2 = int(min(img_w, (x_c + w / 2) * img_w))
            y2 = int(min(img_h, (y_c + h / 2) * img_h))

            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            boxes.append({
                'class_id': cls_id,
                'class_name': cls_name,
                'norm_name': normalize_name(cls_name),
                'bbox': [x1, y1, x2, y2]
            })
    return boxes

def run_validation(delivery_dir, errors_out, candidates_out=None):
    labels_dir = find_file_or_dir(delivery_dir, ['labels', 'det_labels']) or os.path.join(delivery_dir, 'labels')
    panoptic_json_path = find_file_or_dir(delivery_dir, ['panoptic.json', 'annotations/panoptic.json'])
    panoptic_masks_dir = find_file_or_dir(delivery_dir, ['masks', 'panoptic_masks', 'annotations/panoptic_masks'])

    if not panoptic_json_path or not os.path.exists(panoptic_json_path):
        print(f"[-] LỖI: Không tìm thấy panoptic.json trong {delivery_dir}")
        return

    class_names = load_class_names(delivery_dir)

    with open(panoptic_json_path, 'r', encoding='utf-8') as f:
        panoptic_data = json.load(f)

    categories = {int(cat['id']): cat for cat in panoptic_data.get('categories', [])}
    candidates = []

    for ann in panoptic_data.get('annotations', []):
        image_id = ann.get('image_id')
        file_name = ann.get('file_name')
        png_name = os.path.basename(file_name) if file_name else f"{image_id}.png"
        png_path = os.path.join(panoptic_masks_dir, png_name) if panoptic_masks_dir else None

        if not png_path or not os.path.exists(png_path):
            continue

        mask_ids = read_png_id(png_path)
        img_h, img_w = mask_ids.shape

        base_name = os.path.splitext(png_name)[0]
        txt_path = os.path.join(labels_dir, f"{base_name}.txt")
        det_boxes = parse_yolo_txt(txt_path, img_w, img_h, class_names)

        # 1. Chỉ lấy isthing == 1 và iscrowd == 0
        thing_segments = []
        for seg in ann.get('segments_info', []):
            cat_id = int(seg['category_id'])
            cat_info = categories.get(cat_id, {})
            if cat_info.get('isthing', 0) == 1 and seg.get('iscrowd', 0) == 0:
                raw_name = cat_info.get('name', str(cat_id))
                seg['class_name'] = raw_name
                seg['norm_name'] = normalize_name(raw_name)
                thing_segments.append(seg)

        for seg in thing_segments:
            seg_id = seg['id']
            seg_cls_name = seg['class_name']
            seg_norm_name = seg['norm_name']
            seg_bbox_coco = seg['bbox']
            seg_bbox = [
                seg_bbox_coco[0], 
                seg_bbox_coco[1], 
                seg_bbox_coco[0] + seg_bbox_coco[2], 
                seg_bbox_coco[1] + seg_bbox_coco[3]
            ]
            
            seg_pixel_mask = (mask_ids == seg_id)
            segment_area = np.sum(seg_pixel_mask)

            # 2. Bỏ qua segment nhỏ hơn 800 pixels
            if segment_area < 800:
                continue

            evaluated_boxes = []
            for det in det_boxes:
                b = det['bbox']
                sub_mask = seg_pixel_mask[b[1]:b[3], b[0]:b[2]]
                inside_pixels = np.sum(sub_mask)
                
                inside_ratio = inside_pixels / float(segment_area)
                bbox_iou = compute_box_iou(b, seg_bbox)

                evaluated_boxes.append({
                    'det': det,
                    'inside_ratio': inside_ratio,
                    'bbox_iou': bbox_iou
                })

            same_class_boxes = [eb for eb in evaluated_boxes if eb['det']['norm_name'] == seg_norm_name]

            # Lỗi instance_merge
            high_overlap_same = [eb for eb in same_class_boxes if eb['inside_ratio'] >= 0.45 and eb['bbox_iou'] >= 0.35]
            if len(high_overlap_same) >= 2:
                candidates.append({
                    'image_id': image_id,
                    'file_name': png_name,
                    'type': 'instance_merge',
                    'segment_id': seg_id,
                    'class_name': seg_cls_name,
                    'details': f"1 segment mask bị phủ bởi {len(high_overlap_same)} BBox trùng lớp."
                })

            # Có BBox cùng lớp khớp vị trí -> Nhãn đúng
            if same_class_boxes:
                best_same = max(same_class_boxes, key=lambda x: x['bbox_iou'])
                if best_same['bbox_iou'] >= 0.30 or best_same['inside_ratio'] >= 0.40:
                    continue
                elif 0.10 <= best_same['bbox_iou'] < 0.30:
                    candidates.append({
                        'image_id': image_id,
                        'file_name': png_name,
                        'type': 'bbox_drift',
                        'segment_id': seg_id,
                        'class_name': seg_cls_name,
                        'inside_ratio': round(best_same['inside_ratio'], 4),
                        'bbox_iou': round(best_same['bbox_iou'], 4),
                        'details': 'Bounding Box cùng lớp bị lệch so với mask.'
                    })
                    continue

            # Kiểm tra class_swap dựa trên bbox_iou
            diff_class_boxes = [eb for eb in evaluated_boxes if eb['det']['norm_name'] != seg_norm_name]
            if diff_class_boxes:
                best_diff = max(diff_class_boxes, key=lambda x: x['bbox_iou'])
                if best_diff['bbox_iou'] >= 0.50:
                    det = best_diff['det']
                    candidates.append({
                        'image_id': image_id,
                        'file_name': png_name,
                        'type': 'class_swap',
                        'segment_id': seg_id,
                        'det_class': det['class_name'],
                        'seg_class': seg_cls_name,
                        'bbox_iou': round(best_diff['bbox_iou'], 4),
                        'details': f"Lệch nhãn lớp: BBox ({det['class_name']}) vs Panoptic ({seg_cls_name})."
                    })
                    continue

            # Lỗi missing_box
            max_inside = max([eb['inside_ratio'] for eb in evaluated_boxes], default=0)
            max_iou = max([eb['bbox_iou'] for eb in evaluated_boxes], default=0)

            if max_inside < 0.20 and max_iou < 0.20:
                candidates.append({
                    'image_id': image_id,
                    'file_name': png_name,
                    'type': 'missing_box',
                    'segment_id': seg_id,
                    'class_name': seg_cls_name,
                    'details': 'Segment mask không có Bounding Box nào phủ tới.'
                })

    os.makedirs(os.path.dirname(os.path.abspath(errors_out)), exist_ok=True)
    with open(errors_out, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)

    if candidates_out:
        os.makedirs(os.path.dirname(os.path.abspath(candidates_out)), exist_ok=True)
        with open(candidates_out, 'w', encoding='utf-8') as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Đã quét {delivery_dir}: Phát hiện {len(candidates)} ứng viên lỗi.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validator kiểm tra lỗi gán nhãn Dataset Release Handover")
    parser.add_argument('delivery_dir', help="Thư mục delivery cần validation")
    parser.add_argument('--out', required=True, help="Đường dẫn lưu file rules.json/errors.json")
    parser.add_argument('--candidates', help="Đường dẫn lưu file candidates.json (tùy chọn)")
    
    args = parser.parse_args()
    run_validation(args.delivery_dir, args.out, args.candidates)