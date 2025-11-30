import os
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from datasets import load_dataset
import numpy as np

# --- الإعدادات ---
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "./processed_coco_final"
OUTPUT_DIR = "./review_artifacts" # المجلد الذي سنضع فيه أدلة التوثيق
SAMPLES_TO_SAVE = 10 # عدد العينات التي سنحفظها للعرض

def main():
    print("🚀 Starting Tokenization & Embedding Inspection...")
    
    # 1. إنشاء مجلد المخرجات
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. تحميل أدوات المعالجة (Tokenizer & Text Encoder)
    print("⏳ Loading CLIP Tokenizer & Text Encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    
    # 3. تحميل عينة من البيانات
    print(f"📂 Loading dataset from {DATA_DIR}...")
    dataset = load_dataset("imagefolder", data_dir=DATA_DIR, split="train")
    
    # ملف التقرير النصي (ليقرأه البشر)
    report_path = os.path.join(OUTPUT_DIR, "tokenization_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Tokenization & Embedding Inspection Report ===\n")
        f.write(f"Model Used: {MODEL_NAME}\n")
        f.write("=================================================\n\n")
        
        print(f"📝 Processing first {SAMPLES_TO_SAVE} samples...")
        
        for i in range(SAMPLES_TO_SAVE):
            sample = dataset[i]
            original_text = sample["text"]
            image_filename = f"sample_{i}_image.png" # نحفظ الصورة أيضاً للمقارنة
            
            # حفظ الصورة الأصلية
            sample["image"].save(os.path.join(OUTPUT_DIR, image_filename))
            
            # --- الخطوة 1: Tokenization ---
            # تحويل النص إلى أرقام (Input IDs)
            inputs = tokenizer(
                original_text, 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            input_ids = inputs.input_ids
            
            # إعادة فك التشفير (للتأكد أن التوكنز صحيحة)
            decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # --- الخطوة 2: Embeddings ---
            # تمرير الأرقام لموديل اللغة لإنتاج المتجهات
            with torch.no_grad():
                outputs = text_encoder(input_ids)
                last_hidden_state = outputs.last_hidden_state # هذه هي الـ Embeddings
            
            # --- كتابة التقرير ---
            f.write(f"Sample #{i+1}:\n")
            f.write(f"Original Text:  {original_text}\n")
            f.write(f"Token IDs:      {input_ids[0].numpy().tolist()[:15]} ... (truncated)\n") # نعرض أول 15 رقم فقط
            f.write(f"Decoded Check:  {decoded_text}\n")
            f.write(f"Embedding Shape: {last_hidden_state.shape}  <-- (Batch, Sequence Length, Vector Dim)\n")
            f.write("-" * 50 + "\n")
            
            # --- حفظ الـ Embeddings كملف تقني (.pt) ---
            # هذا الملف تثبت به أنك استخرجت المتجهات فعلياً
            embedding_path = os.path.join(OUTPUT_DIR, f"sample_{i}_embedding.pt")
            torch.save(last_hidden_state, embedding_path)

    print("\n✅ DONE! Artifacts saved.")
    print(f"📄 Report File: {os.path.abspath(report_path)}")
    print(f"📂 Embeddings Tensors: {os.path.abspath(OUTPUT_DIR)}")
    print("💡 You can open 'tokenization_report.txt' now to review the steps.")

if __name__ == "__main__":
    main()