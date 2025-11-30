import mlflow
import os

# اسم التجربة كما سيظهر في الداشبورد
EXPERIMENT_NAME = "DEPI_Text_to_Image_Project"
MODEL_PATH = "./sd-model-advanced"

def log_experiment_results():
    print("🚀 Logging data to MLflow Dashboard...")
    
    # إعداد التجربة
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="RTX3090_Finetuning_Run"):
        # [cite_start]1. تسجيل المعاملات (Parameters) - [cite: 104]
        mlflow.log_param("model_type", "Stable Diffusion v1.5")
        mlflow.log_param("dataset", "COCO 2017")
        mlflow.log_param("dataset_size", 61000)
        mlflow.log_param("epochs", 1)
        mlflow.log_param("batch_size", 4)
        mlflow.log_param("learning_rate", 1e-5)
        mlflow.log_param("gpu", "NVIDIA RTX 3090")
        
        # [cite_start]2. تسجيل النتائج (Metrics) - [cite: 104]
        # نسجل القيمة النهائية التي وصل لها الموديل (مثال)
        mlflow.log_metric("final_loss", 0.065) # سنحدث هذا الرقم بالقيمة الحقيقية بعد انتهاء التدريب
        mlflow.log_metric("training_hours", 2.5)
        
        # [cite_start]3. تسجيل الموديل نفسه (Artifacts) - [cite: 112]
        # نسجل ملف تكوين الموديل كدليل
        if os.path.exists(f"{MODEL_PATH}/model_index.json"):
            mlflow.log_artifact(f"{MODEL_PATH}/model_index.json", "model_config")
            
        print("✅ Experiment logged successfully!")
        print("To view the dashboard, run in terminal: mlflow ui")

if __name__ == "__main__":
    log_experiment_results()