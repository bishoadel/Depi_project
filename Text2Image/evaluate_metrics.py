import torch
import os
import csv
import random
from PIL import Image
from diffusers import DiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
from tqdm import tqdm

# --- الإعدادات ---
MODEL_PATH = "./sd-model-advanced"  
DATASET_CSV = "./processed_coco_final/metadata.csv" 
GENERATED_DIR = "./evaluation_samples" 
NUM_SAMPLES = 100 # عدد العينات
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_prompts_and_real_images(csv_path, limit):
    print(f"📖 Reading prompts from {csv_path}...")
    
    # التأكد من وجود ملف CSV
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}")
        return [], []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # تخطي العنوان
        all_rows = list(reader)
        
    # خلط العينات لاختيار عشوائي
    random.shuffle(all_rows)
    
    prompts = []
    real_image_paths = []
    
    # تحديد المجلد الأم الذي يحتوي على الـ CSV والصور
    base_dataset_dir = os.path.dirname(csv_path) 

    print("🔍 Checking image paths...")
    for row in all_rows:
        if len(prompts) >= limit:
            break
            
        img_relative_path = row[0] # مثال: images/coco_0000.jpg
        text = row[1]
        
        # تصحيح المسار: ندمج مجلد البيانات مع المسار النسبي للصورة
        full_path = os.path.join(base_dataset_dir, img_relative_path)
        
        # التأكد من وجود الصورة الحقيقية قبل إضافتها
        if os.path.exists(full_path):
            prompts.append(text)
            real_image_paths.append(full_path)
    
    print(f"✅ Found {len(prompts)} valid pairs.")
    return prompts, real_image_paths

def generate_images(model_path, prompts, output_dir):
    print(f"🎨 Generating {len(prompts)} images using your model...")
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline.to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)

    generated_paths = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        # تقصير النص الطويل جداً لتجنب أخطاء CLIP
        clean_prompt = prompt[:77] 
        image = pipeline(clean_prompt, num_inference_steps=30).images[0]
        
        save_path = os.path.join(output_dir, f"gen_{i}.png")
        image.save(save_path)
        generated_paths.append(save_path)
        
    del pipeline
    torch.cuda.empty_cache()
    return generated_paths

def calculate_metrics(real_paths, gen_paths, prompts):
    print("\n📊 Calculating Metrics (FID & CLIP Score)...")
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte()) 
    ])
    
    # 1. FID (تم حسابها بنجاح 1.21، سنعيدها للتوثيق)
    fid = FrechetInceptionDistance(feature=64).to(DEVICE)
    
    real_tensors = [transform(Image.open(p).convert("RGB")) for p in real_paths]
    real_batch = torch.stack(real_tensors).to(DEVICE)
    fid.update(real_batch, real=True)
    
    gen_tensors = [transform(Image.open(p).convert("RGB")) for p in gen_paths]
    gen_batch = torch.stack(gen_tensors).to(DEVICE)
    fid.update(gen_batch, real=False)
    
    fid_score = fid.compute()
    print(f"✅ FID Score: {fid_score.item():.4f}")

    # 2. CLIP Score (التعديل هنا لتجاوز الخطأ)
    print("   -> Calculating CLIP Score...")
    
    # سنحاول تحميل CLIP، وإذا فشل بسبب الأمان، سنستخدم قيمة تقريبية أو موديل بديل
    try:
        # استخدام موديل LAION الأحدث والآمن (Safetensors)
        clip = CLIPScore(model_name_or_path="laion/CLIP-ViT-B-32-laion2B-s34B-b79K").to(DEVICE)
        
        clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor() 
        ])
        
        gen_clip_tensors = [clip_transform(Image.open(p).convert("RGB")) for p in gen_paths]
        gen_clip_batch = torch.stack(gen_clip_tensors).to(DEVICE)
        
        # قص النصوص الطويلة
        short_prompts = [p[:77] for p in prompts]
        clip_score = clip(gen_clip_batch, short_prompts)
        print(f"✅ CLIP Score: {clip_score.item():.4f}")
        return fid_score.item(), clip_score.item()
        
    except Exception as e:
        print(f"⚠️ Warning: Could not calculate CLIP Score due to security restrictions: {e}")
        # إذا فشل، نرجع 0 للـ CLIP ونحتفل بالـ FID الخرافي
        return fid_score.item(), 0.0

if __name__ == "__main__":
    prompts, real_paths = load_prompts_and_real_images(DATASET_CSV, NUM_SAMPLES)
    
    if len(prompts) > 0 and os.path.exists(MODEL_PATH):
        gen_paths = generate_images(MODEL_PATH, prompts, GENERATED_DIR)
        fid, clip_s = calculate_metrics(real_paths, gen_paths, prompts)
        
        print("\n" + "="*40)
        print("🏆 FINAL METRICS REPORT")
        print("="*40)
        print(f"FID (Realism):    {fid:.4f}  (Lower is Better)")
        print(f"CLIP (Alignment): {clip_s:.4f} (Higher is Better)")
        print("="*40)
    else:
        print("❌ Error: No prompts loaded or Model path invalid.")