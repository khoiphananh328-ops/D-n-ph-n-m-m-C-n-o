from google import genai
import os


# ==========================================
# LẤY API KEY
# ==========================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "❌ Chưa có GEMINI_API_KEY.\n"
        "Hãy đặt API key trong Terminal trước."
    )


# ==========================================
# KẾT NỐI GEMINI
# ==========================================

print("🔄 Đang kết nối Gemini...")

client = genai.Client(
    api_key=api_key
)

print("✅ Đã kết nối Gemini")
print("🤖 Đang gửi câu hỏi...")


# ==========================================
# GỌI GEMINI
# ==========================================

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=(
        "Bạn là trợ lý AI cho hệ thống "
        "'Sinh vật biển và bảo tồn sinh vật biển tại Côn Đảo'. "
        "Hãy giới thiệu ngắn gọn về đa dạng sinh học biển Côn Đảo."
    )
)


# ==========================================
# HIỂN THỊ KẾT QUẢ
# ==========================================

print()
print("=" * 50)
print("🤖 GEMINI AI")
print("=" * 50)

print(response.text)

print("=" * 50)
print("✅ TEST GEMINI THÀNH CÔNG")
print("=" * 50)