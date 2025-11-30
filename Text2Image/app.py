import gradio as gr
import torch
from diffusers import DiffusionPipeline

# --- 1. إعدادات الموديل ---
MODEL_PATH = "./sd-model-advanced"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"⏳ Loading Model on {device}... (This may take a moment)")

try:
    # محاولة تحميل الموديل المدرب
    pipe = DiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    pipe.to(device)
    status_msg = "✅ Using Your Finetuned Model (COCO 2017)"
except Exception as e:
    print(f"⚠️ Error loading finetuned model: {e}")
    print("⚠️ Switching to Base Model for demo purposes...")
    # تحميل الموديل الأصلي كبديل
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.to(device)
    status_msg = "⚠️ Using Base Model (Training might not be fully saved yet)"

# --- 2. دالة التوليد ---
def generate_image(prompt, negative_prompt, steps, guidance):
    if not prompt:
        return None
    
    # التوليد
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=int(steps), 
        guidance_scale=guidance
    ).images[0]
    
    return image

# --- 3. تصميم الواجهة (النسخة القياسية المضمونة) ---
# أزلنا أي إعدادات للثيمات لضمان التوافق مع أي إصدار
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 DEPI Generative AI Project")
    gr.Markdown(f"### {status_msg}")
    
    with gr.Row():
        with gr.Column():
            # المدخلات
            txt_prompt = gr.Textbox(label="Enter your prompt", placeholder="e.g., a futuristic city on mars...", lines=2)
            txt_negative = gr.Textbox(label="Negative Prompt", value="low quality, blurry, distorted", lines=1)
            
            # إعدادات متقدمة
            with gr.Accordion("Advanced Settings", open=False):
                slider_steps = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Inference Steps")
                slider_guidance = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale")
            
            # زر التشغيل
            btn_generate = gr.Button("🚀 Generate Image")
            
        with gr.Column():
            # المخرجات
            output_img = gr.Image(label="Generated Result", type="pil")

    # ربط الزر بالدالة
    btn_generate.click(generate_image, inputs=[txt_prompt, txt_negative, slider_steps, slider_guidance], outputs=output_img)

# تشغيل التطبيق
if __name__ == "__main__":
    print("🌐 Starting Web UI...")
    demo.launch(share=True)