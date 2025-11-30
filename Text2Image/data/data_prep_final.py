import os
import requests
import zipfile
import json
import csv
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import concurrent.futures

# --- الإعدادات ---
OUTPUT_DIR = "./processed_coco_final"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
TARGET_SIZE = (512, 512)

# 🔥 تم التعديل: 60 ألف صورة
LIMIT = 118000 
NUM_THREADS = 16 

# روابط سيرفرات COCO الرسمية
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_BASE_URL = "http://images.cocodataset.org/train2017/"

def create_directories():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    print(f"✅ Directories ready: {OUTPUT_DIR}")

def download_file(url, filename):
    """تحميل ملف مع شريط تقدم"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_and_process_single_image(args):
    """دالة المعالجة مع ميزة الاستكمال الذكي"""
    img_info, caption, idx = args
    file_name = img_info['file_name']
    img_url = IMAGES_BASE_URL + file_name
    
    save_name = f"coco_{idx:06d}.jpg"
    save_path = os.path.join(IMAGE_DIR, save_name)
    clean_text = caption.lower().strip()

    # --- ميزة الاستكمال الذكي ---
    # إذا كانت الصورة موجودة مسبقاً وسليمة، لا نحملها مرة أخرى
    if os.path.exists(save_path):
        return [save_name, clean_text] 
    # ---------------------------

    try:
        # 1. تحميل
        response = requests.get(img_url, timeout=10)
        if response.status_code != 200:
            return None

        img_bytes = BytesIO(response.content)
        image = Image.open(img_bytes)

        # 2. معالجة
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

        # 3. حفظ
        image.save(save_path, quality=95)
        
        return [save_name, clean_text]

    except Exception as e:
        return None

def main():
    print(f"🚀 Starting High-Volume Downloader (Target: {LIMIT})...")
    create_directories()

    # 1. التعليقات (Annotations)
    anno_zip = "annotations.zip"
    # التحقق مما إذا كان الملف موجوداً بالفعل لتوفير التحميل
    if not os.path.exists("annotations/captions_train2017.json"):
        if not os.path.exists(anno_zip):
            print("📥 Downloading Annotations...")
            download_file(ANNOTATIONS_URL, anno_zip)
        
        print("📦 Extracting Annotations...")
        with zipfile.ZipFile(anno_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
    else:
        print("✅ Annotations already found, skipping download.")
    
    # 2. قراءة البيانات
    print("📖 Reading JSON metadata...")
    with open('annotations/captions_train2017.json', 'r') as f:
        coco_data = json.load(f)
    
    images_dict = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']

    tasks = []
    print(f"📋 Preparing list for {LIMIT} images...")
    
    processed_img_ids = set()
    count = 0
    
    for ann in annotations:
        img_id = ann['image_id']
        if img_id in processed_img_ids:
            continue 
        
        if img_id in images_dict:
            tasks.append((images_dict[img_id], ann['caption'], count))
            processed_img_ids.add(img_id)
            count += 1
        
        if count >= LIMIT:
            break

    # 3. التشغيل
    print(f"🔥 Starting Download ({NUM_THREADS} threads)...")
    print("Existing images will be skipped automatically.")
    
    with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            results = list(tqdm(executor.map(download_and_process_single_image, tasks), total=len(tasks), unit="img"))
            
            for res in results:
                if res:
                    writer.writerow(res)

    print(f"\n✅ MISSION ACCOMPLISHED!")
    print(f"📊 Total Images: {LIMIT}")
    print(f"📂 Saved at: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()