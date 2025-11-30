import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DConditionModel, DiffusionPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from tqdm.auto import tqdm
import logging
import math

# --- إعدادات التدريب ---
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "./sd-model-finetuned"
DATA_DIR = "./processed_coco_final"

# إعدادات الـ GPU
TRAIN_BATCH_SIZE = 4 
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1 

# =========================================================
# 🔥 إصلاح ويندوز: تعريف أدوات المعالجة خارج الـ Main
# =========================================================

# 1. تعريف الـ Tokenizer عالمياً
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")

# 2. تعريف تحويلات الصور عالمياً
train_transforms = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# 3. دالة المعالجة أصبحت عالمية (Global Function)
def preprocess_train(examples):
    # معالجة الصور
    images = [train_transforms(image.convert("RGB")) for image in examples["image"]]
    
    # معالجة النصوص
    text_inputs = tokenizer(
        examples["text"], 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    return {
        "pixel_values": images,
        "input_ids": text_inputs.input_ids,
    }
# =========================================================


def main():
    # 1. إعداد المسرع
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="fp16" 
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print("\n" + "="*40)
        print(f"🚀 STARTING PROJECT: Text-to-Image Generation")
        print(f"🖥️  GPU: RTX 3090 Detected (Windows Fix Applied)")
        print("="*40 + "\n")

    # 2. تحميل الموديلات الثقيلة (داخل Main للحفاظ على الذاكرة)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

    # التجميد (Freeze)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # 3. تحميل البيانات
    if accelerator.is_main_process:
        print(f"⏳ Loading Dataset from {DATA_DIR}...")

    dataset = load_dataset("imagefolder", data_dir=DATA_DIR, split="train")

    # تطبيق دالة المعالجة (التي أصبحت global الآن)
    with accelerator.main_process_first():
        train_dataset = dataset.with_transform(preprocess_train)

    # DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2
    )

    # 4. المجهز (Optimizer)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    # التحضير
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    
    text_encoder.to(accelerator.device, dtype=torch.float16)
    vae.to(accelerator.device, dtype=torch.float16)

    # حساب الخطوات
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch

    # 5. حلقة التدريب
    if accelerator.is_main_process:
        print(f"\n✅ Training Started (Total Steps: {max_train_steps})... GO! 👇")

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Training Progress", unit="step")

    for epoch in range(NUM_EPOCHS):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # تحويل الصور Latents
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # إضافة الضوضاء
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # النص
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # التنبؤ
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # الخسارة
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # التحديث
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix({"loss": f"{loss.detach().item():.4f}"})

    # 6. الحفظ
    if accelerator.is_main_process:
        print("\n⏳ Saving Final Model...")
        pipeline = DiffusionPipeline.from_pretrained(
            MODEL_NAME,
            unet=accelerator.unwrap_model(unet),
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
        )
        pipeline.save_pretrained(OUTPUT_DIR)
        print("\n" + "="*40)
        print(f"🎉 MISSION ACCOMPLISHED!")
        print(f"📂 Model saved at: {os.path.abspath(OUTPUT_DIR)}")
        print("="*40)

if __name__ == "__main__":
    main()