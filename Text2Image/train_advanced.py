import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DConditionModel, DiffusionPipeline, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from tqdm.auto import tqdm
import logging
import math

# --- إعدادات التدريب المتقدم ---
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "./sd-model-advanced"  # سنحفظ الموديل الجديد في مجلد منفصل
DATA_DIR = "./processed_coco_final" # مجلد البيانات الضخم الخاص بك

# --- Hyperparameters ---
TRAIN_BATCH_SIZE = 6    # رفعنا الباتش لاستغلال الـ 24GB VRAM
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3          # 3 دورات كاملة (جودة عالية)
LR_SCHEDULER = "cosine" # نظام ذكي لتقليل معدل التعلم تدريجياً
LR_WARMUP_STEPS = 500

# تعريف الأدوات (Global for Windows Fix)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")

# --- تحسين المعالجة (Augmentation) ---
# إضافة RandomHorizontalFlip لزيادة تنوع البيانات
train_transforms = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(512),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def preprocess_train(examples):
    images = [train_transforms(image.convert("RGB")) for image in examples["image"]]
    text_inputs = tokenizer(
        examples["text"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    return {"pixel_values": images, "input_ids": text_inputs.input_ids}

def main():
    # 1. إعداد المسرع
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="fp16"
    )

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print("\n" + "="*50)
        print(f"🚀 STARTING ADVANCED TRAINING (Epochs: {NUM_EPOCHS})")
        print(f"📂 Dataset: {DATA_DIR} (126k Images)")
        print(f"🧠 Scheduler: {LR_SCHEDULER} | Augmentation: ON")
        print("="*50 + "\n")

    # 2. تحميل الموديلات
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

    # تفعيل Gradient Checkpointing لتوفير الذاكرة مع الباتش الكبير
    unet.enable_gradient_checkpointing()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # 3. تحميل البيانات
    if accelerator.is_main_process:
        print("⏳ Loading Ultimate Dataset...")
        
    dataset = load_dataset("imagefolder", data_dir=DATA_DIR, split="train")
    
    with accelerator.main_process_first():
        train_dataset = dataset.with_transform(preprocess_train)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    # حساب الخطوات والـ Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS * accelerator.num_processes,
        num_training_steps=max_train_steps,
    )

    # التحضير
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    text_encoder.to(accelerator.device, dtype=torch.float16)
    vae.to(accelerator.device, dtype=torch.float16)

    # 4. التدريب
    if accelerator.is_main_process:
        print(f"✅ Ready! Total Optimization Steps: {max_train_steps}")
        print("🚀 Training Started... (This will take hours, monitor your GPU temps!)")

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Advanced Training", unit="step")

    for epoch in range(NUM_EPOCHS):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                train_loss += loss.detach().item()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix({"loss": f"{train_loss / (step + 1):.4f}"})

        # حفظ نسخة (Checkpoint) بعد كل Epoch
        if accelerator.is_main_process:
            epoch_save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
            print(f"\n💾 Saving Checkpoint for Epoch {epoch+1}...")
            pipeline = DiffusionPipeline.from_pretrained(
                MODEL_NAME,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_encoder,
                vae=vae,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(epoch_save_path)

    # الحفظ النهائي
    if accelerator.is_main_process:
        pipeline.save_pretrained(OUTPUT_DIR)
        print("\n" + "="*50)
        print(f"🎉 ADVANCED TRAINING COMPLETE!")
        print(f"📂 Final Model: {os.path.abspath(OUTPUT_DIR)}")
        print("="*50)

if __name__ == "__main__":
    main()