import os
from pathlib import Path
from io import BytesIO

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Côn Đảo Conservation — phiên bản giao diện web chuyển từ Tkinter
BASE_DIR = Path(__file__).resolve().parent
# Đọc API key từ file .env khi chạy trên máy cá nhân.
load_dotenv(BASE_DIR / ".env")
MODEL_PATH = BASE_DIR / "best.pt"
ASSETS_DIR = BASE_DIR / "assests"  # Giữ đúng tên thư mục trong ứng dụng gốc

BG = "#03151F"
HEADER = "#062633"
PANEL = "#092F3C"
CYAN = "#45E7FF"
GREEN = "#45E59A"
YELLOW = "#E8D66E"
WHITE = "#F1FBFF"
MUTED = "#91AFB8"

st.set_page_config(
    page_title="Côn Đảo | Marine Conservation",
    page_icon="🐢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
:root {{ color-scheme: dark; }}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background: {BG} !important; color: {WHITE} !important;
}}
.block-container {{ padding-top: 1.4rem; max-width: 1400px; }}
/* Force high-contrast text across Streamlit widgets and markdown. */
.stApp, .stApp p, .stApp li, .stApp label, .stApp span,
.stApp div[data-testid="stMarkdownContainer"], .stApp div[data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"],
.stApp [data-testid="stFileUploader"] {{ color: {WHITE} !important; }}
.stApp [data-testid="stCaptionContainer"] p {{ color: #C5D9E0 !important; }}
[data-testid="stHeader"] {{ background: rgba(3,21,31,.98) !important; }}
[data-testid="stSidebar"] {{ background: {HEADER} !important; }}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span {{ color: #F4FBFF !important; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: #C5D9E0 !important; }}
.hero {{ background: linear-gradient(120deg, #062633, #0C3947); border: 1px solid #247084;
         border-radius: 22px; padding: 2rem; margin-bottom: 1rem; }}
.hero h1 {{ color: {CYAN} !important; margin-bottom: .25rem; }}
.hero p {{ color: #F0FAFC !important; font-size: 1.05rem; }}
.panel {{ background: {PANEL}; border: 1px solid #247084; border-radius: 16px; padding: 1.1rem; }}
.small-muted {{ color: #C5D9E0 !important; }}
/* Alerts: readable text regardless of Streamlit's default alert theme. */
[data-testid="stAlert"] {{ border-radius: 12px; }}
[data-testid="stAlert"] p, [data-testid="stAlert"] div,
[data-testid="stAlert"] span {{ color: #F7FBFF !important; }}
[data-testid="stAlert"] a {{ color: #8EDCFF !important; }}
[data-testid="stProgress"] {{ color: {CYAN}; }}
div.stButton > button {{ border-radius: 10px; border: 1px solid #45A5BD;
    color: #FFFFFF !important; background: #087EA4; font-weight: 600; }}
div.stButton > button:hover {{ background: #0A91BB; border-color: #7CEBFF; }}
[data-testid="stFileUploaderDropzone"] {{ background: #0A2B38; border: 1px dashed #5DAFC3; }}
[data-testid="stFileUploaderDropzone"] * {{ color: #F4FBFF !important; }}
input, textarea {{ color: #F4FBFF !important; }}
hr {{ border-color: #24505C !important; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Đang tải mô hình YOLO, lần đầu có thể mất một lúc...")
def load_yolo(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)

@st.cache_resource(show_spinner=False)
def load_gemini_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)

def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return (key or os.environ.get("GEMINI_API_KEY", "")).strip()

def ask_gemini(image_bytes: bytes, detections, client):
    if not detections:
        return "Không phát hiện sinh vật bằng YOLO. Hãy thử ảnh rõ hơn hoặc chụp gần đối tượng hơn."
    from google import genai
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((768, 768), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=80, optimize=True)
    yolo_text = "\n".join(f"- {name}: {confidence * 100:.1f}%" for name, confidence in detections)
    prompt = f"""Bạn là trợ lý AI hỗ trợ bảo tồn sinh vật tại Côn Đảo.
YOLO đã nhận diện:\n{yolo_text}
Hãy cung cấp thông tin ngắn gọn bằng tiếng Việt:
1. Tên sinh vật
2. Đặc điểm chính
3. Môi trường sống
4. Vai trò sinh thái
5. Khuyến nghị bảo tồn
Mỗi mục chỉ 1-2 câu. Không cần nhận diện lại loài nếu YOLO đã cung cấp tên. Không khẳng định tuyệt đối nếu ảnh không đủ rõ."""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[prompt, genai.types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")],
    )
    return response.text or "Gemini chưa trả về nội dung."

def analyze(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Ảnh được tải lên", use_container_width=True)
    try:
        if not MODEL_PATH.exists():
            st.error("Chưa tìm thấy file best.pt. Hãy đặt best.pt cùng thư mục với app.py và tải lại ứng dụng.")
            return
        model = load_yolo(str(MODEL_PATH))
        with st.spinner("YOLO đang nhận diện sinh vật..."):
            results = model.predict(source=Image.open(BytesIO(image_bytes)).convert("RGB"), conf=0.15, verbose=False)
        result = results[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append((model.names[class_id], confidence))
        detections.sort(key=lambda item: item[1], reverse=True)

        st.subheader("🤖 Kết quả nhận diện YOLO")
        if detections:
            for name, confidence in detections:
                st.write(f"**{name}** — độ tin cậy {confidence * 100:.1f}%")
            st.progress(max(0.0, min(1.0, detections[0][1])))
        else:
            st.warning("Không phát hiện sinh vật trong ảnh.")

        api_key = get_api_key()
        st.subheader("✨ Phân tích Gemini AI")
        if not api_key:
            st.warning("Chưa cấu hình GEMINI_API_KEY. YOLO vẫn có thể nhận diện; thêm API key trong Secrets để bật phần phân tích Gemini.")
            return
        try:
            client = load_gemini_client(api_key)
            with st.spinner("Gemini đang phân tích đặc điểm và khuyến nghị bảo tồn..."):
                answer = ask_gemini(image_bytes, detections, client)
            st.markdown(answer)
        except Exception as e:
            error_text = str(e)
            st.error(f"Gemini chưa phân tích được ảnh: {error_text}")
            st.caption("Kiểm tra file .env có đúng tên GEMINI_API_KEY, API key còn hiệu lực, model được cấp quyền và hạn mức API. Không chia sẻ API key công khai.")
    except Exception as e:
        st.error(f"Không thể phân tích ảnh: {e}")

with st.sidebar:
    st.markdown("# 🐢 CÔN ĐẢO")
    st.caption("MARINE CONSERVATION")
    page = st.radio("ĐIỀU HƯỚNG", ["Trang chủ & AI", "Bảo tồn", "Giới thiệu"])
    st.divider()
    st.markdown("🟢 **AI SYSTEM**")
    st.caption("YOLO nhận diện sinh vật • Gemini hỗ trợ cung cấp thông tin")

st.markdown("""
<div class="hero">
  <h1>🌊 CÔN ĐẢO CONSERVATION</h1>
  <p>Ứng dụng trí tuệ nhân tạo hỗ trợ nhận diện sinh vật và nâng cao nhận thức bảo tồn hệ sinh thái biển Côn Đảo.</p>
</div>
""", unsafe_allow_html=True)

if page == "Trang chủ & AI":
    col1, col2, col3 = st.columns(3)
    col1.metric("Hệ thống nhận diện", "YOLO AI")
    col2.metric("Phân tích thông tin", "Gemini AI")
    col3.metric("Mục tiêu", "Bảo tồn biển")
    st.markdown("## 🔎 Nhận diện sinh vật biển")
    st.write("Tải lên ảnh JPG, JPEG, PNG hoặc WEBP để hệ thống thử nhận diện sinh vật bằng mô hình YOLO và nhận thông tin hỗ trợ từ Gemini.")
    uploaded = st.file_uploader("Chọn ảnh sinh vật", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if uploaded is not None:
        if st.button("🚀 Phân tích ảnh", type="primary", use_container_width=True):
            analyze(uploaded)
        else:
            st.image(uploaded, caption="Ảnh đã chọn — bấm Phân tích ảnh để bắt đầu", use_container_width=True)
    st.info("Lưu ý: kết quả AI chỉ mang tính tham khảo; cần chuyên gia xác minh khi dùng cho mục đích nghiên cứu hoặc bảo tồn.")

elif page == "Bảo tồn":
    st.markdown("## 🛡️ BẢO TỒN SINH VẬT VÀ HỆ SINH THÁI BIỂN CÔN ĐẢO")
    st.caption("Thông tin chuyển từ nội dung phần Bảo tồn trong ứng dụng gốc.")
    sections = [
        ("🌊 Hệ sinh thái biển Côn Đảo", "Công cuộc bảo tồn hệ sinh thái và tài nguyên biển Côn Đảo được khởi xướng sau khi thành lập Vườn quốc gia Côn Đảo. Công tác theo dõi, giám sát với sự tham gia của chuyên gia khoa học và cán bộ Vườn quốc gia giúp nhận diện các thách thức và định hướng bảo tồn trong tương lai."),
        ("🪸 Rạn san hô", "Rạn san hô chịu tác động lớn từ bão Linda (11/1997) và sự cố tẩy trắng san hô năm 1998. Đến năm 2005, độ phủ san hô cứng trung bình chỉ trên 23,9%; đến năm 2019 rạn san hô phục hồi gần như nguyên trạng. Hiện tượng tẩy trắng tiếp tục xảy ra vào các năm 2010, 2016 và 2019 do nhiệt độ nước biển tầng mặt tăng. Một số khu vực chịu tác động của con người có độ phủ thấp, như khoảng 30% ở Cựa Gà và 15% ở Bãi Cát Lớn."),
        ("🌱 Thảm cỏ biển và Bò biển", "Thảm cỏ biển có ý nghĩa quan trọng về sinh thái và là nguồn sống của Bò biển. Tài liệu gốc ghi nhận độ phủ thảm cỏ biển có xu hướng suy giảm mạnh, chỉ bằng khoảng 1/10 so với năm 2009–2010; đồng thời có dấu hiệu suy thoái như phủ trầm tích và rong phụ sinh. Cần tăng cường quản lý chất thải, đặc biệt là chất thải hữu cơ."),
        ("🐟 Sinh vật rạn san hô", "Mật độ cá Bướm tăng cùng với sự cải thiện độ phủ san hô, trong khi một số loài có giá trị thực phẩm suy giảm hoặc duy trì ở mật độ thấp. Mật độ nhóm cá Hồng được ghi nhận giảm từ 80 xuống 20 con/100 m²; mật độ Trai Tai tượng giảm 4 lần trong giai đoạn 2005–2019. Phục hồi Trai Tai tượng, Ốc đụn, Bào ngư và san hô có ý nghĩa sinh thái và kinh tế."),
        ("🐢 Bảo tồn Rùa biển", "Trong giai đoạn 2009–2019, trung bình gần 1.500 tổ Rùa xanh được di dời chăm sóc mỗi năm, tương ứng hơn 137.000 trứng và gần 110.000 rùa con nở. Các chỉ số này tăng xấp xỉ gấp đôi so với giai đoạn 1998–2008. Tỷ lệ rùa nở và di chuyển ra biển sau đó khá ổn định, khoảng 75–85% tổng số trứng."),
        ("🎯 Định hướng bảo tồn đến năm 2030", "• Bảo tồn hiệu quả đa dạng hệ sinh thái biển, nhất là rạn san hô và thảm cỏ biển.\n\n• Bảo vệ và phục hồi các loài quý hiếm, bị đe dọa, chú trọng Bò biển, Rùa biển, Trai Tai tượng, Ốc đụn cái, Trai ngọc môi vàng và Cầu gai đá.\n\n• Phục hồi san hô cứng; hỗ trợ hệ sinh thái thích ứng với biến đổi khí hậu; giảm tác động tiêu cực của hoạt động kinh tế – xã hội; tăng cường nghiên cứu, giám sát và sự tham gia của cộng đồng."),
    ]
    for title, body in sections:
        with st.container(border=True):
            st.markdown(f"### {title}")
            st.write(body)
    st.caption("Nguồn được ghi trong ứng dụng gốc: tài liệu ‘Thách thức, kỳ vọng – Vườn quốc gia Côn Đảo’.")

elif page == "Giới thiệu":
    st.title("GIỚI THIỆU VỀ CÔN ĐẢO")

    st.markdown("""
    Côn Đảo là vùng đất có giá trị đặc biệt về lịch sử, văn hóa và đa dạng sinh học.
    Không chỉ nổi tiếng với những dấu ấn lịch sử hào hùng, nơi đây còn sở hữu
    hệ sinh thái rừng và biển phong phú, có ý nghĩa quan trọng đối với công tác
    bảo tồn thiên nhiên tại Việt Nam.
    """)

    # 1. VỊ TRÍ VÀ TỰ NHIÊN
    st.header("1. Vị trí và điều kiện tự nhiên")

    st.subheader("Vị trí địa lý")
    st.markdown("""
    Côn Đảo nằm ngoài khơi vùng biển phía Nam Việt Nam, cách:
    - Vũng Tàu khoảng 185 km.
    - Thành phố Hồ Chí Minh khoảng 230 km.
    - Cần Thơ khoảng 83 km.

    Côn Đảo có vị trí địa lý đặc biệt, góp phần hình thành nên những
    giá trị tự nhiên và hệ sinh thái đặc trưng của vùng biển phía Nam.
    """)

    st.subheader("Địa hình và khí hậu")
    st.markdown("""
    Khoảng 2/3 diện tích Côn Đảo là rừng núi, với đỉnh cao nhất là
    Núi Thánh Giá, cao khoảng 577 m.

    Khí hậu mang tính chất nhiệt đới gió mùa – hải dương, được chia
    thành hai mùa rõ rệt:
    - **Mùa mưa**
    - **Mùa khô**

    Nhiệt độ trung bình năm khoảng 26°C.
    """)

    st.subheader("Hệ sinh thái")
    st.markdown("""
    Côn Đảo sở hữu hệ sinh thái tự nhiên đa dạng, bao gồm hệ sinh thái
    rừng nhiệt đới, rừng ngập mặn, thảm cỏ biển và các rạn san hô.

    **Vườn quốc gia Côn Đảo** là khu vực có vai trò quan trọng trong
    việc bảo tồn đa dạng sinh học. Nơi đây nổi tiếng là địa điểm sinh sản
    của nhiều loài rùa biển và là nơi sinh sống của nhiều loài động,
    thực vật quý hiếm.

    Vườn quốc gia Côn Đảo được công nhận là Khu Ramsar và Vườn Di sản ASEAN,
    khẳng định giá trị sinh thái và tầm quan trọng quốc tế của khu vực này.
    """)

    st.markdown(
        "[Tìm hiểu thêm về Vườn quốc gia Côn Đảo](https://www.condaopark.com.vn/vn/tong-quan-vuon-quoc-gia-con-dao.html)"
    )

    st.divider()

    # 2. LỊCH SỬ
    st.header("2. Lịch sử Côn Đảo")

    st.markdown("""
    Côn Đảo từng được biết đến với tên gọi **“địa ngục trần gian”**,
    gắn liền với hệ thống Nhà tù Côn Đảo được thành lập từ năm 1862.
    Trong hơn một thế kỷ, nơi đây từng là nơi giam giữ nhiều chiến sĩ
    cách mạng và những người yêu nước Việt Nam.

    Những dấu tích lịch sử tại Côn Đảo là minh chứng cho tinh thần
    kiên cường, bất khuất và sự hy sinh của các thế hệ đi trước.

    Ngày nay, Côn Đảo trở thành địa chỉ đỏ trong các hoạt động giáo dục
    truyền thống, tham quan lịch sử và du lịch tâm linh. Trong đó,
    nghĩa trang Hàng Dương và phần mộ Anh hùng liệt sĩ Võ Thị Sáu
    là những địa điểm được nhiều người đến viếng thăm.
    """)

    st.divider()

    # 3. DU LỊCH VÀ ẨM THỰC
    st.header("3. Du lịch và ẩm thực")

    st.subheader("Vẻ đẹp tự nhiên")
    st.markdown("""
    Côn Đảo nổi bật với cảnh quan thiên nhiên hoang sơ, làn nước biển
    trong xanh, những bãi cát trắng và các rạn san hô có giá trị sinh thái.

    Nơi đây từng được nhiều tạp chí du lịch quốc tế bình chọn trong
    các danh sách điểm đến đảo hấp dẫn. Theo thông tin du lịch bạn
    cung cấp, thời điểm đẹp để du lịch biển thường là từ **tháng 3 đến tháng 9**.
    """)

    st.subheader("Ẩm thực đặc trưng")
    st.markdown("""
    Ẩm thực Côn Đảo mang nét đặc trưng của vùng biển đảo với nhiều
    món ăn và đặc sản địa phương, tiêu biểu như:

    - **Ốc vú nàng:** đặc sản hải sản nổi tiếng của Côn Đảo.
    - **Tôm mũ ni:** loại hải sản được nhiều du khách yêu thích.
    - **Mực một nắng:** món ăn đặc trưng của vùng biển.
    - **Mứt hạt bàng:** đặc sản được chế biến từ hạt bàng Côn Đảo.
    - **Rượu sâm đất:** một sản phẩm đặc trưng được giới thiệu tại địa phương.
    """)

    st.divider()

    # 4. GIÁ TRỊ CỦA CÔN ĐẢO ĐỐI VỚI CÔNG TÁC BẢO TỒN
    st.header("4. Giá trị của Côn Đảo đối với công tác bảo tồn")

    st.markdown("""
    Với sự phong phú về hệ sinh thái và sự hiện diện của nhiều loài
    sinh vật quý hiếm, Côn Đảo là địa điểm có ý nghĩa quan trọng
    trong nghiên cứu khoa học và bảo tồn đa dạng sinh học.

    Việc ứng dụng trí tuệ nhân tạo vào nhận diện và giám sát sinh vật
    có thể góp phần hỗ trợ thu thập thông tin, nhận biết các loài,
    nâng cao hiệu quả theo dõi đa dạng sinh học và cung cấp thêm
    dữ liệu phục vụ công tác bảo tồn tại địa phương.
    """)

    st.info(
        "Côn Đảo không chỉ là vùng đất giàu giá trị lịch sử, "
        "văn hóa mà còn là một khu vực có ý nghĩa đặc biệt đối với "
        "công tác bảo tồn thiên nhiên và đa dạng sinh học."
    )

    # NGUỒN THAM KHẢO
    with st.expander("Nguồn tài liệu tham khảo"):
        st.markdown("""
        1. [MIA.vn – Du lịch Côn Đảo](https://mia.vn/cam-nang-du-lich/du-lich-con-dao-13220)
        2. [Cổng thông tin Côn Đảo – Thông tin tổng quan](https://condao.com.vn/vi/news/Tin-tuc/thong-tin-tong-quan-huyen-con-dao-1676.html)
        3. [Vườn quốc gia Côn Đảo](https://www.condaopark.com.vn/vn/tong-quan-vuon-quoc-gia-con-dao.html)
        4. [VnExpress – Cẩm nang du lịch Côn Đảo](https://vnexpress.net/cam-nang-du-lich-con-dao-4445727.html)
        5. [Côn Đảo Explorer](https://condaoexplorer.com/con-dao)
        6. [Umove Adventure – Tất tần tật về Côn Đảo](https://www.umoveadventure.com/vi/tat-tan-tat-ve-con-dao.html)
        7. [Wikipedia tiếng Việt – Côn Đảo](https://vi.wikipedia.org/wiki/C%C3%B4n_%C4%90%E1%BA%A3o)
        """)