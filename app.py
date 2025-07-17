# app.py
from flask import Flask, request, jsonify
from pythainlp.tokenize import word_tokenize # Để tách từ
from pythainlp.tag import pos_tag # Để gán từ loại
# from pythainlp.tag import perceptron # Một số POS tagger cần được import cụ thể

app = Flask(__name__)

# --- Hàm tìm từ đồng nghĩa/trái nghĩa (ví dụ từ trước) ---
def find_related_thai_words(word):
    mock_data = {
        "สวัสดี": {
            "synonyms": ["หวัดดี", "เรียน"],
            "antonyms": [],
            "related_words": ["สบายดีไหม", "ขอบคุณ"]
        },
        "ใหญ่": {
            "synonyms": ["โต", "มหึมา"],
            "antonyms": ["เล็ก"],
            "related_words": ["สูง", "กว้าง"]
        },
        "ดี": {
            "synonyms": ["เยี่ยม", "เลิศ"],
            "antonyms": ["ไม่ดี"],
            "related_words": ["เก่ง", "ถูกต้อง"]
        }
    }
    result = mock_data.get(word, {"synonyms": [], "antonyms": [], "related_words": []})
    return result

@app.route('/tra-tu-lien-quan', methods=['POST'])
def tra_tu_lien_quan():
    data = request.get_json()
    word = data.get('word', '').strip()

    if not word:
        return jsonify({"error": "Vui lòng cung cấp từ cần tra cứu."}), 400

    results = find_related_thai_words(word)
    return jsonify(results)

# --- ENDPOINT MỚI: TÌM TỪ LOẠI ---
@app.route('/get-pos-tag', methods=['POST'])
def get_pos_tag():
    data = request.get_json()
    text = data.get('text', '').strip() # Đặt tên biến là 'text' vì có thể là cả câu

    if not text:
        return jsonify({"error": "Vui lòng cung cấp văn bản để tìm từ loại."}), 400

    try:
        # Tách từ
        words = word_tokenize(text)
        
        # Gán từ loại (sử dụng POS tagger mặc định của PyThaiNLP, thường là Perceptron)
        # Nếu gặp lỗi, bạn có thể cần tải mô hình bằng cách chạy:
        # from pythainlp.tag import pos_tag
        # pos_tag.download('perceptron') # Hoặc các mô hình khác như 'artagger', 'best'
        # Hoặc ensure mô hình được tải trong môi trường triển khai.
        tagged_words = pos_tag(words) # Kết quả là list of tuples: [('word', 'POS_TAG'), ...]
        
        # Chuyển đổi sang định dạng dễ dùng hơn cho JSON
        pos_results = [{"word": w, "pos": p} for w, p in tagged_words]
        
        return jsonify({"original_text": text, "pos_tags": pos_results})

    except Exception as e:
        return jsonify({"error": f"Lỗi khi xử lý từ loại: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Chỉ dùng khi phát triển cục bộ
