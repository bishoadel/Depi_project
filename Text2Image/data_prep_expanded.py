import os
import requests
import zipfile
import json
import csv
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import concurrent.futures
import shutil

# --- الإعدادات الموسعة ---
OUTPUT_DIR = "./processed_coco_final"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
TARGET_SIZE = (512, 512)
NUM_THREADS = 16 

# 1. إعدادات COCO (سنحمل كل الـ 118 ألف صورة)
COCO_LIMIT = 118000 
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGES_BASE_URL = "http://images.cocodataset.org/train2017/"

# 2. إعدادات Flickr8k (روابط مباشرة مستقرة)
FLICKR_IMAGES_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
FLICKR_TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

def create_directories():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    print(f"✅ Directories ready: {OUTPUT_DIR}")

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"✅ File {filename} already exists. Skipping download.")
        return
    
    print(f"📥 Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
        desc=filename, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

# --- معالجة COCO ---
def process_coco_image(args):
    img_info, caption, idx = args
    file_name = img_info['file_name']
    img_url = COCO_IMAGES_BASE_URL + file_name
    
    save_name = f"images/coco_{idx:06d}.jpg" # لاحظ: أضفنا images/ للمسار مباشرة
    full_save_path = os.path.join(OUTPUT_DIR, save_name)
    
    # تخطي الموجود
    if os.path.exists(full_save_path):
        return [save_name, caption.lower().strip()]

    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code != 200: return None
        
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        image.save(full_save_path, quality=95)
        
        return [save_name, caption.lower().strip()]
    except:
        return None

# --- معالجة Flickr ---
def process_flickr_data():
    print("\n🚀 Starting Flickr8k Integration...")
    
    # تحميل الملفات
    download_file(FLICKR_IMAGES_URL, "Flickr8k_Dataset.zip")
    download_file(FLICKR_TEXT_URL, "Flickr8k_text.zip")
    
    # فك الضغط
    if not os.path.exists("Flicker8k_Dataset"):
        print("📦 Extracting Flickr Images...")
        with zipfile.ZipFile("Flickr8k_Dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
            
    if not os.path.exists("Flickr8k.token.txt"):
        print("📦 Extracting Flickr Text...")
        with zipfile.ZipFile("Flickr8k_text.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
            
    # قراءة النصوص
    flickr_mapping = {}
    with open("Flickr8k.token.txt", "r") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) < 2: continue
            img_id = parts[0].split("#")[0]
            caption = parts[1].strip()
            # نأخذ أول تعليق فقط لكل صورة لتجنب التكرار
            if img_id not in flickr_mapping:
                flickr_mapping[img_id] = caption

    print(f"📋 Found {len(flickr_mapping)} Flickr images to process.")
    
    processed_entries = []
    source_dir = "Flicker8k_Dataset" # اسم المجلد بعد فك الضغط قد يختلف قليلاً حسب المصدر
    
    # تصحيح اسم المجلد الشائع
    if not os.path.exists(source_dir) and os.path.exists("Flickr8k_Dataset"):
        source_dir = "Flickr8k_Dataset"

    for img_file, caption in tqdm(flickr_mapping.items(), desc="Processing Flickr"):
        src_path = os.path.join(source_dir, img_file)
        save_name = f"images/flickr_{img_file}"
        dst_path = os.path.join(OUTPUT_DIR, save_name)
        
        if os.path.exists(dst_path):
            processed_entries.append([save_name, caption.lower()])
            continue
            
        if os.path.exists(src_path):
            try:
                image = Image.open(src_path).convert("RGB")
                image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                image.save(dst_path, quality=95)
                processed_entries.append([save_name, caption.lower()])
            except:
                continue
                
    return processed_entries

def main():
    create_directories()
    
    all_metadata = []

    # ---------------- PART 1: COCO ----------------
    print(f"🚀 PART 1: Processing COCO ({COCO_LIMIT} images)...")
    
    # تحميل Annotations
    if not os.path.exists("annotations/captions_train2017.json"):
        download_file(COCO_ANNOTATIONS_URL, "annotations.zip")
        with zipfile.ZipFile("annotations.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
            
    with open('annotations/captions_train2017.json', 'r') as f:
        coco_data = json.load(f)
        
    images_dict = {img['id']: img for img in coco_data['images']}
    tasks = []
    processed_ids = set()
    
    for ann in coco_data['annotations']:
        if len(tasks) >= COCO_LIMIT: break
        if ann['image_id'] not in processed_ids and ann['image_id'] in images_dict:
            tasks.append((images_dict[ann['image_id']], ann['caption'], len(tasks)))
            processed_ids.add(ann['image_id'])

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(tqdm(executor.map(process_coco_image, tasks), total=len(tasks), desc="COCO Download"))
        for res in results:
            if res: all_metadata.append(res)

    # ---------------- PART 2: FLICKR ----------------
    flickr_entries = process_flickr_data()
    all_metadata.extend(flickr_entries)

    # ---------------- SAVE FINAL CSV ----------------
    print(f"\n💾 Saving merged metadata for {len(all_metadata)} total images...")
    with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        writer.writerows(all_metadata)

    print(f"\n✅ ULTIMATE DATASET READY!")
    print(f"📂 Location: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📊 Total Images: {len(all_metadata)} (COCO + Flickr)")

if __name__ == "__main__":
    main()