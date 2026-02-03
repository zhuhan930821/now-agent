import pdfplumber
import re

def clean_pdf_to_txt(pdf_path, txt_path):
    text_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # --- 1. 切除页眉页脚 (Crop) ---
            # 假设页眉在顶部 50px，页脚在底部 50px，我们只取中间部分
            # bbox = (x0, top, x1, bottom)
            width = page.width
            height = page.height
            
            # 这里根据你的 PDF 实际情况调整参数
            # 例如：切掉顶部 60 像素，底部 60 像素
            cropped_page = page.crop((0, 60, width, height - 60))
            
            text = cropped_page.extract_text()
            
            if text:
                # --- 2. 清洗常见噪音 ---
                # 去除页码（如果是单纯数字占一行）
                text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
                # 修复被截断的单词 (如 Power- ful -> Powerful)
                text = text.replace('-\n', '')
                # 去除多余换行，把段落连起来
                text = text.replace('\n', ' ')
                
                text_content.append(text)

    # 保存为干净的 txt
    full_text = "\n\n".join(text_content)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"清洗完成！已保存为: {txt_path}")

# 运行
clean_pdf_to_txt("data/powerofnow.pdf", "data/cleaned_book.txt")