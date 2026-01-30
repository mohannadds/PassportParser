import ollama
from pathlib import Path
import sys
import base64
import time


IMAGE_PATH = r"C:\Users\User\Downloads\IMG-20250518-WA0005.jpg"
MODEL = "qwen3-vl:4b"


def main():
    print(f"--- Fast Passport OCR ({MODEL}) ---")
    
    path = Path(IMAGE_PATH)
    if not path.exists():
        print(f"[ERROR] Image not found: {path}")
        return

    print(f"[1/3] Loading image: {path.name}...")
    try:
        with open(path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            print(f"      -> Image loaded ({len(image_base64)} chars base64)")
    except Exception as e:
        print(f"[ERROR] Failed to read image: {e}")
        return

    prompt = """Extract data from this passport image.
    Return ONLY a JSON object with these fields:
    surname, given_names, nationality, date_of_birth, sex, 
    place_of_birth, date_of_issue, date_of_expiry, passport_number, mrz_lines.
    
    If a field is not visible, use null.
    Do NOT write any introduction or conclusion. ONLY the JSON."""


    print(f"[2/3] Sending to Ollama (streaming)...")
    print("-" * 40)
    
    full_response = ""
    start_time = time.time()
    
    try:
        stream = ollama.chat(
            model=MODEL,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_base64]
            }],
            stream=True
        )
        
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            full_response += content
            
    except Exception as e:
        print(f"\n[ERROR] Ollama call failed: {e}")
        print("Tip: Make sure Ollama is running and the model is pulled ('ollama pull qwen3-vl:4b')")
        return

    duration = time.time() - start_time
    print(f"\n\n" + "-" * 40)
    print(f"[3/3] Done in {duration:.2f}s")
    
    # 4. Save Output
    output_filename = path.stem + "_fast_output.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_response)
    print(f"[INFO] Saved raw output to: {output_filename}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    main()

