import streamlit as st
import cv2
import csv
import tempfile
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import easyocr
import time
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from IPython.display import display, Image

groq_api_key = st.secrets["groq"]["groq_api_key"]
llm = ChatGroq(
    model='llama3-70b-8192',
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=groq_api_key
) 
## 
DRIVING_ASSISTANT_PROMPT_TEMPLATE = """
### تعليمات:
أنت مساعد ذكي متقدم ومتخصص في استخدام النطام باللغة العربية. مهمتك هي إرشاد مستخدمين النظام حول كيفيه استخدام النظام. لقد تم تدريبي على ما يلي:

**متطلبات الستخدم عند استخدام نماذج الذكاء الصناعي:**
• عند ارفاق مقطع فديو لنموذج, فأنه يجب الا يتعدى 100 ميقا بايت.
• ان يكون الفيديو مصور بأستخدام الداش كام.
• أن يكون موقع الداش كام في المركبه بمنتصف الزجاج الأمامي.
• ان يكون المقطع المصور ببيئة صالحه للقيادة.
• عند استخدام نموذج السرعه, يجب اضهار سرعه المركبة في المفطع.

**عندما يرحب بك المستخدم, رد الترحيب**

# نظام تحليل مخالفات المرور عبر الفيديو

## خطوات استخدام النظام:

1. **تسجيل الدخول:**
   - قم بتسجيل الدخول إلى النظام من خلال الرابط المتاح.

2. **رفع الفيديو:**
   - اختر الفيديو المراد رفعه، بشرط أن يكون مصورًا باستخدام كاميرا السيارة الأمامية.
   - تأكد من أن حجم الفيديو لا يتجاوز 100 ميغا بايت، وأن البيئة المصورة صالحة للقيادة.

3. **معالجة الفيديو:**
   - انتظر حتى يتم معالجة الفيديو وعرض النتائج.

4. **عرض النتائج:**
   - استعرض النتائج التي تشمل:
     - عدد المخالفات المتعلقة بالسرعة.
     - عدد مخالفات عدم التوقف عند الإشارات.
     - المخالفات المتعلقة بالمسافة.

5. **تنزيل النتائج:**
   - بإمكانك تنزيل النتائج النهائية كملف CSV للمراجعة والتقارير.


في البداية رحب بالمستخدم واظهر له خيارين مساعدين الاول هو عن ما اذا اراد معرفة متطلبات النظام او فائدة النظام وكيف يزيد من السلامة العامة.
اعتمد على هذه المعلومات للإجابة على استفسارات المستخدم حول طريقة استخدام النظام.
اشرح باختصار عن فائدة النظام باجاز.
في حال كان السؤال غير واضح أو خارج نطاق هذه المعلومات، يرجى طلب التوضيح أو الرد بأدب بأنك لا تستطيع الإجابة.

**إجابتك يجب أن تكون باللغة العربية**

**if he asks a question related to about driving, provide him with the website for the General Department of Traffic 'https://www.moi.gov.sa' placeholder for the website: General Department of Traffic**

---
### السؤال: {subject}
---
### الإجابة:
"""

# Define prompt template for the chatbot
prompt = PromptTemplate(
    input_types={'subject': 'string', 'language': 'string'},
    template=DRIVING_ASSISTANT_PROMPT_TEMPLATE
)

# Initialize the chain for the chatbot
chain = LLMChain(llm=llm, prompt=prompt)


# Function to process Stop Sign Model
def process_stop_sign_model(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    MOTION_THRESHOLD = 0.75
    STOP_SIGN_CLASS_ID = 0
    REQUIRED_STOP_FRAMES = fps
    output_path = '/tmp/output_stop_sign.avi'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_gray = None
    stopped_frame_count = 0
    stop_sign_detected = False
    stop_completed = False
    stop_sign_id = 0
    sign_logged = False
    stop_sign_detected_frame_count = 0

    total_stop_signs = 0
    total_stopped = 0
    total_not_stopped = 0

    results = []
    csv_file_path = '/tmp/stop_sign_results.csv'
    header = ['stop_sign_id', 'stopped']

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = model.track(frame, persist=True, conf=0.7, show=False)
            stop_sign_in_frame = False

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f'{model.names[int(class_id)]}: {conf:.2f}'

                    if int(class_id) == STOP_SIGN_CLASS_ID:
                        stop_sign_in_frame = True
                        stop_sign_detected_frame_count += 1

                        if not stop_sign_detected and stop_sign_detected_frame_count >= REQUIRED_STOP_FRAMES:
                            stop_sign_id += 1
                            stop_sign_detected = True
                            sign_logged = False
                            total_stop_signs += 1

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if not stop_completed and prev_gray is not None:
                            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            flow_magnitude = np.linalg.norm(flow, axis=2).mean()
                            if flow_magnitude < MOTION_THRESHOLD:
                                stopped_frame_count += 1
                                cv2.putText(frame, "Vehicle Stopped", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            else:
                                stopped_frame_count = 0

                            if stopped_frame_count >= REQUIRED_STOP_FRAMES:
                                cv2.putText(frame, "Stop Completed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                stop_completed = True

            if stop_sign_detected and not stop_sign_in_frame and not sign_logged:
                if stop_completed:
                    total_stopped += 1
                else:
                    total_not_stopped += 1
                writer.writerow({'stop_sign_id': stop_sign_id, 'stopped': stop_completed})
                stop_sign_detected = False
                stop_completed = False
                stopped_frame_count = 0
                stop_sign_detected_frame_count = 0

            prev_gray = gray.copy()
            out.write(frame)

        cap.release()
        out.release()

    return {
        'total_stop_signs': total_stop_signs,
        'total_stopped': total_stopped,
        'total_not_stopped': total_not_stopped,
        'csv_file_path': csv_file_path,
        'output_path': output_path
    }

# Function to process Speed Model
def process_speed_model(video_path, model, reader):
    def preprocess_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
        return cv2.equalizeHist(denoised_image)

    def enhance_image(image):
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty. Cannot enhance.")
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return blurred

    def resize_image(image, scale_factor=2):
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    def extract_region(image, box):
        x1, y1, x2, y2 = map(int, box)
        height, width = image.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            return None
        return image[y1:y2, x1:x2]

    def read_text_with_easyocr(image):
        preprocessed_image = enhance_image(image)
        if preprocessed_image is None or preprocessed_image.size == 0:
            st.warning("Preprocessed image is empty. Skipping OCR.")
            return []
        resized_image = resize_image(preprocessed_image)
        results = reader.readtext(resized_image)
        return [(text, prob) for (bbox, text, prob) in results if text.isdigit()]

    def detect_speed_signs(video_path, confidence_threshold=0.5, frame_interval=60, ocr_interval=10):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video.")
            return 0

        frame_count = 0
        speed_violations = 0
        last_detected_speed_sign_text = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                results = model.track(frame, persist=True, show=False)
                best_conf, best_box = 0, None

                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    for i, box in enumerate(boxes):
                        conf = confidences[i]
                        if conf > confidence_threshold and int(result.boxes.cls[i]) == 0:
                            if conf > best_conf:
                                best_conf = conf
                                best_box = box

                if best_box is not None:
                    speed_sign_region = extract_region(frame, best_box)
                    if speed_sign_region is not None and speed_sign_region.size > 0:
                        speed_sign_texts = read_text_with_easyocr(speed_sign_region)
                        if speed_sign_texts:
                            speed_sign_text = ''.join([text for text, prob in speed_sign_texts])
                            if speed_sign_text:
                                last_detected_speed_sign_text = speed_sign_text

            if frame_count % ocr_interval == 0 and last_detected_speed_sign_text:
                fixed_box = (850, 950, 50, 90)  # Adjust coordinates as needed
                fixed_box_region = extract_region(frame, fixed_box)
                if fixed_box_region is not None and fixed_box_region.size > 0:
                    fixed_box_texts = read_text_with_easyocr(fixed_box_region)
                    fixed_box_text = ''.join([text for text, prob in fixed_box_texts])

                    if fixed_box_text and int(fixed_box_text) > int(last_detected_speed_sign_text):
                        speed_violations += 1

            frame_count += 1

        cap.release()
        return speed_violations

    violations = detect_speed_signs(video_path)
    return violations

# Function to process Distance Model
def process_distance_model(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video for distance model processing.")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    center_x = width // 2
    offset = int(0.1 * height)
    offset1 = int(0.25 * height)
    green_dot = (center_x, height - offset1)
    red_dot = (center_x, height - offset)

    violation_count = 0
    violated_cars = set()
    violation = 0

    # Initialize previous frame size
    prev_frame_size = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Verify frame size
        current_frame_size = frame.shape[:2]  # (height, width)
        if prev_frame_size is None:
            prev_frame_size = current_frame_size
        else:
            if current_frame_size != prev_frame_size:
                st.warning("Frame size mismatch detected. Resizing frame to match previous frames.")
                frame = cv2.resize(frame, (prev_frame_size[1], prev_frame_size[0]))
            else:
                pass  # Sizes match, no action needed

        try:
            results = model.track(frame, persist=True, show=False)
        except cv2.error as e:
            st.error(f"OpenCV error during tracking: {e}")
            break
        except Exception as e:
            st.error(f"Unexpected error during tracking: {e}")
            break

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 2:  # Assuming 'car' class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    car_bottom = (int((x1 + x2) / 2), y2)
                    car_id = f"{x1}{y1}{x2}_{y2}"
                    if green_dot[1] <= car_bottom[1] < red_dot[1]:
                        if car_id not in violated_cars:
                            violated_cars.add(car_id)
                    elif car_bottom[1] >= red_dot[1]:
                        violation += 1
                        violation_count = int((violation / 100) / 2)
        # st.write(f"Processing... Current violations detected: {violation_count}")

    cap.release()
    return violation_count

# Function to load models
@st.cache_resource
def load_models():
    stop_model = YOLO("models/StopModel.pt")
    speed_model = YOLO('models/SpeedModel.pt')
    distance_model = YOLO('models/DistanceModel.pt')
    reader = easyocr.Reader(['en'])
    return stop_model, speed_model, distance_model, reader

stop_model, speed_model, distance_model, reader = load_models()

# Function to append results to summary CSV
def append_summary_csv(summary_csv_path, city, not_stopped, speed_violations, distance_violations):
    file_exists = os.path.isfile(summary_csv_path)
    with open(summary_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['run_timestamp', 'city','stop_sign_violations', 'speed_violations', 'distance_violations']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'run_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'city':city,
            'stop_sign_violations': not_stopped,
            'speed_violations': speed_violations,
            'distance_violations': distance_violations,
        })

# Define path for the summary CSV
SUMMARY_CSV_PATH = 'summary_results.csv'

# Streamlit App Layout

# Streamlit App Layout
st.sidebar.title("Navigation")

if st.sidebar.button("Admin Dashboard"):
    st.session_state.page = "Admin Dashboard"

if st.sidebar.button("User Dashboard"):
    st.session_state.page = "User Dashboard"

if st.sidebar.button("ChatBot"):
    st.session_state.page = "ChatBot"

if st.sidebar.button("Model"):
    st.session_state.page = "Run All Models"

if 'page' not in st.session_state:
    st.session_state.page = "User Dashboard"

#    ---- ChatBot ----   
if st.session_state.page == "ChatBot":
    st.title("ChatBot")
    st.write("مرحبا")

    # Display the welcome message
    welcome_message = """كيف يمكنني مساعدتك؟ :\n
    • مثال: كيفية استخدام النظام؟
    • متطلبات رفع فيديو؟

    """
    st.markdown(welcome_message)

    # Initialize session state for conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages in the conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input section
    if user_input := st.chat_input("Ask your driving question:"):
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user's message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process response from ChatGroq
        subject = user_input
        language = "Arabic"
        response = chain.run(subject=subject, language=language)

        # Display assistant's response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Append the assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

#    -------   models  ------
if st.session_state.page == "Run All Models":
    st.title("AI Traffic Video Analysis")
    cities = ['Other', 'Riyadh', 'Jeddah', 'Dammam', 'Makkah', 'Madinah']
    city = st.selectbox("Select the city that the video recorded in:", cities)
    st.write("Videos for testing: https://drive.google.com/drive/folders/1-bp2_C9PCtEpOyEzaAWX2sjthRIGJCTq?usp=sharing")

    uploaded_file = st.file_uploader("Upload your driving video (max size: 100MB)", type=["mp4", "mov", "avi"])

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.write(f"Uploaded file size: {file_size_mb:.2f} MB")

        if file_size_mb > 100:
            st.error("File too large! Please upload a file smaller than 100MB.")
        else:
            st.video(uploaded_file)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(uploaded_file.read())
                video_path = temp_video_file.name

            # Initialize variables to collect metrics
            not_stopped = 0
            speed_violations = 0
            distance_violations = 0

            # Stop Sign Model
            with st.spinner("Processing Stop Sign Model..."):
                stop_results = process_stop_sign_model(video_path, stop_model)
                st.success("Stop Sign Model processed successfully!")
                not_stopped = stop_results['total_not_stopped']

            # Speed Model
            with st.spinner("Processing Speed Model..."):
                speed_violations = process_speed_model(video_path, speed_model, reader)
                st.success("Speed Model processed successfully!")

            # Distance Model
            with st.spinner("Processing Distance Model..."):
                distance_violations = process_distance_model(video_path, distance_model)
                st.success("Distance Model processed successfully!")

            st.balloons()
            st.success("All models have been processed successfully!")

            # Append results to summary CSV
            append_summary_csv(SUMMARY_CSV_PATH, city, not_stopped, speed_violations, distance_violations)

            # Create a summary table using pandas
            summary_data = {
                'Violation Type': ['Cars Not Stopped at Stop Signs', 'Speed Violations', 'Distance Violations'],
                'Count': [not_stopped, speed_violations, distance_violations]
            }
            df_summary = pd.DataFrame(summary_data)

            # Display the table of violations
            st.table(df_summary)

            # Provide download button for summary CSV
            if os.path.exists(SUMMARY_CSV_PATH):
                with open(SUMMARY_CSV_PATH, 'rb') as f:
                    st.download_button(
                        label="Download Summary Results CSV",
                        data=f,
                        file_name="summary_results.csv",
                        mime="text/csv"
                    )


# Check if 'page' state exists and is set to "Admin Dashboard"
import pandas as pd
import streamlit as st
import datetime

if st.session_state.page == "Admin Dashboard":
    st.header("Traffic Violations Dashboard")

    # Load data
    df = pd.read_csv("summary_results.csv")
    # Ensure 'run_timestamp' is datetime
    df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])

    # Time slicer for filtering data (between 2023 and 2026)
    min_date = datetime.date(2023, 1, 1)
    max_date = datetime.date(2026, 12, 31)

    start_date, end_date = st.slider(
        "Select Date Range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Filter data based on the time range selected
    filtered_data = df[(df['run_timestamp'] >= pd.to_datetime(start_date)) & (df['run_timestamp'] <= pd.to_datetime(end_date))]

    # Create a selection box for cities
    all_cities = ['All'] + filtered_data['city'].unique().tolist()  # Add 'All' option
    selected_city = st.selectbox("Select City:", all_cities, index=0)

    # Further filter the data based on selected city
    if selected_city != 'All':
        filtered_data = filtered_data[filtered_data['city'] == selected_city]

    # Calculate summary statistics for the selected date range and city
    cars_not_stopped = filtered_data['stop_sign_violations'].sum()
    speed_violations = filtered_data['speed_violations'].sum()
    distance_violations = filtered_data['distance_violations'].sum()

    # Display the metrics in boxes
    col1, col2, col3 = st.columns(3)
    col1.metric("Cars Not Stopped at Stop Signs", cars_not_stopped)
    col2.metric("Speed Violations", speed_violations)
    col3.metric("Distance Violations", distance_violations)

    # Line chart for the selected date range
    st.subheader("Violations Over Time")
    chart_data = filtered_data.set_index('run_timestamp')[['stop_sign_violations', 'speed_violations', 'distance_violations']]
    st.line_chart(chart_data)

if st.session_state.page == "User Dashboard":
    # Load the data
    df = pd.read_csv("summary_results.csv")

    # Ensure 'run_timestamp' is in datetime format
    df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])
    # Get the last row (KPI) based on 'run_timestamp'
    last_run = df.iloc[-1]

    # Dashboard: Display KPI (the last run data)
    st.header("User Dashboard")

    # Display the last run KPI in 3 boxes (current values for the latest run)
    col1, col2, col3 = st.columns(3)
    col1.metric("Cars Not Stopped (Last Run)", last_run['stop_sign_violations'])
    col2.metric("Speed Violations (Last Run)", last_run['speed_violations'])
    col3.metric("Distance Violations (Last Run)", last_run['distance_violations'])

    # Calculate the total violations
    total_cars_not_stopped = df['stop_sign_violations'].sum()
    total_speed_violations = df['speed_violations'].sum()
    total_distance_violations = df['distance_violations'].sum()

    # Display total violations in boxes
    st.subheader("Total Violations Overview")
    col1_total, col2_total, col3_total = st.columns(3)
    col1_total.metric("Total Cars Not Stopped at Stop Signs", total_cars_not_stopped)
    col2_total.metric("Total Speed Violations", total_speed_violations)
    col3_total.metric("Total Distance Violations", total_distance_violations)

    # Line chart showing the violations over time
    st.subheader("Violations Over Time")
    chart_data = df.set_index('run_timestamp')[['stop_sign_violations', 'speed_violations', 'distance_violations']]
    st.line_chart(chart_data)


