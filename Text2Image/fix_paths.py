import csv
import os
import shutil

# المسار الحالي للملف
csv_file = "./processed_coco_final/metadata.csv"
temp_file = "./processed_coco_final/metadata_temp.csv"

print("🔧 Fixing CSV paths...")

with open(csv_file, 'r', encoding='utf-8') as infile, \
     open(temp_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # قراءة الهيدر (file_name, text)
    header = next(reader)
    writer.writerow(header)
    
    count = 0
    for row in reader:
        filename = row[0]
        text = row[1]
        
        # إذا لم يكن المسار يحتوي على images/ نقوم بإضافتها
        if not filename.startswith("images/"):
            filename = f"images/{filename}"
        
        writer.writerow([filename, text])
        count += 1

# استبدال الملف القديم بالجديد
os.replace(temp_file, csv_file)

print(f"✅ Success! Fixed paths for {count} images.")
print("Now the dataset loader will look inside the 'images' folder correctly.")