import streamlit as st
import cv2
import csv
import temp
import tempfile
import numpy as np
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain_groq import ChatGroq
from ultralytics import YOLO
import easyocr
from IPython.display import display, Image

# Create buttons for each page
if st.sidebar.button("ChatBot"):
    st.session_state.page = "Welcome"
if st.sidebar.button("Stop Sign Model"):
    st.session_state.page = "Stop Sign Model"
if st.sidebar.button("Speed Model"):
    st.session_state.page = "Speed Model"
if st.sidebar.button("Distance Model"):
    st.session_state.page = "Distance Model"
# Initialize ChatGroq LLM
groq_api_key = st.secrets["groq"]["groq_api_key"]
llm = ChatGroq(
    model='llama3-70b-8192',
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=groq_api_key
) 
## the website that i got those instruction is from General Department of Traffic website -> Traffic Safety -> Safe Drive
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

# Sidebar for navigation
st.sidebar.title("Navigation")
# Initialize a session state variable for page tracking
if 'page' not in st.session_state:
    st.session_state.page = "Welcome"


# Welcome Page with Chatbot
if st.session_state.page == "Welcome":
    st.title("ChatBot")
    st.write("مرحبا")

    # Display the welcome message
    welcome_message = """كيف يمكنني مساعدتك؟ :\n
    • مثال: ماهي طريقة القيادة الآمنة والصحيحة"""
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

# Stop Sign Model Logic
elif st.session_state.page == "Stop Sign Model":
    st.title("Stop Sign Model")

    # Allow users to upload a video with a size limit
    uploaded_file = st.file_uploader("Upload your driving video (max size: 100MB)", type=["mp4", "mov", "avi"])

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)  # Convert size to MB
        st.write(f"Uploaded file size: {file_size_mb:.2f} MB")

        if file_size_mb > 100:  # Check if the file size exceeds 100 MB
            st.error("File too large! Please upload a file smaller than 100MB.")
        else:
            # Display the uploaded video
            st.video(uploaded_file)

            # Load the stop sign model using YOLO
            model = YOLO("models/StopModel.pt")  # Use YOLO directly

            # Read the uploaded video using OpenCV
            video_bytes = uploaded_file.read()
            video_path = "/tmp/uploaded_video.mp4"  # Temporarily save the video
            with open(video_path, 'wb') as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define stop model parameters
            MOTION_THRESHOLD = 0.75  # Adjust as necessary
            STOP_SIGN_CLASS_ID = 0  # Stop sign class
            REQUIRED_STOP_FRAMES = fps  # 1 second stop
            output_path = '/tmp/output_video.avi'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            prev_gray = None
            stopped_frame_count = 0
            stop_sign_detected = False
            stop_completed = False
            stop_sign_id = 0
            sign_logged = False
            stop_sign_detected_frame_count = 0

            # Counters for results
            total_stop_signs = 0
            total_stopped = 0
            total_not_stopped = 0

            results = []

            # CSV storage
            csv_file_path = '/tmp/stop_sign_results.csv'
            header = ['stop_sign_id', 'stopped']

            with open(csv_file_path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writeheader()
                

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.write('Processing finished or no more frames to read.')
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    results = model.track(frame, persist=True, conf=0.7)  # Get results from the model
                    stop_sign_in_frame = False
                    
                    for result in results:
                        boxes = result.boxes.xyxy.cpu().numpy()  # Access the 'boxes' attribute
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
                                    total_stop_signs += 1  # Increment total stop signs encountered

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                if not stop_completed and prev_gray is not None:
                                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                    flow_magnitude = np.linalg.norm(flow, axis=2).mean()
                                    print(f'Flow Magnitude: {flow_magnitude}') # just debugging
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
                        # Increment stopped or not stopped counter

                    prev_gray = gray.copy()
                    out.write(frame)

                cap.release()
                out.release()
                
                # Display results
                st.write(f"Total Stop Signs Encountered: {total_stop_signs}")
                st.write(f"Total Times Stopped: {total_stopped}")
                st.write(f"Total Times Not Stopped: {total_not_stopped}")

                st.write(f"Results saved at: {csv_file_path}")
                st.success("Video analyzed successfully!")
# Speed Model Logic

elif st.session_state.page == "Speed Model":
    st.title("Speed Model")
    reader = easyocr.Reader(['en'])

    # File upload section with size check
    uploaded_file = st.file_uploader("Upload your driving video for speed detection (max size: 100MB)", type=["mp4", "mov", "avi"])

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)  # Convert file size to MB
        st.write(f"Uploaded file size: {file_size_mb:.2f} MB")

        if file_size_mb > 100:
            st.error("File too large! Please upload a file smaller than 100MB.")
        else:
            # Display the uploaded video
            st.video(uploaded_file)

            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(uploaded_file.read())
                video_path = temp_video_file.name

            # Load the YOLO speed detection model
            model = YOLO('models/SpeedModel.pt')

            # Preprocessing functions
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
                # Ensure the box is within the image bounds
                height, width = image.shape[:2]
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)

                if x1 >= x2 or y1 >= y2:
                    return None  # Return None if the region is invalid
                return image[y1:y2, x1:x2]

            def read_text_with_easyocr(image):
                preprocessed_image = enhance_image(image)
                if preprocessed_image is None or preprocessed_image.size == 0:
                    print("Warning: preprocessed_image is None or empty. Skipping OCR.")
                    return []
                resized_image = resize_image(preprocessed_image)
                results = reader.readtext(resized_image)
                return [(text, prob) for (bbox, text, prob) in results if text.isdigit()]

            # Speed sign detection function
            def detect_speed_signs(video_path, confidence_threshold=0.5, frame_interval=60, ocr_interval=10):
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Error: Could not open video.")
                    return None

                frame_count = 0
                speed_violations = 0
                last_detected_speed_sign_text = None  # Store last detected speed sign

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
                                if conf > confidence_threshold and int(result.boxes.cls[i]) == 0:  # Speed sign detected
                                    if conf > best_conf:
                                        best_conf = conf
                                        best_box = box

                        if best_box is not None:
                            # Extract the region of the speed sign and read text
                            speed_sign_region = extract_region(frame, best_box)
                            if speed_sign_region is not None and speed_sign_region.size > 0:
                                speed_sign_texts = read_text_with_easyocr(speed_sign_region)
                                if speed_sign_texts:
                                    speed_sign_text = ''.join([text for text, prob in speed_sign_texts])
                                    if speed_sign_text:
                                        st.write(f"Speed sign detected at frame {frame_count}: {speed_sign_text}")
                                        last_detected_speed_sign_text = speed_sign_text

                    # Check for speed violations at fixed intervals
                    if frame_count % ocr_interval == 0 and last_detected_speed_sign_text:
                        fixed_box = (850, 950, 50, 90)  # Adjust coordinates as needed
                        fixed_box_region = extract_region(frame, fixed_box)
                        if fixed_box_region is not None and fixed_box_region.size > 0:
                            fixed_box_texts = read_text_with_easyocr(fixed_box_region)
                            fixed_box_text = ''.join([text for text, prob in fixed_box_texts])

                            if fixed_box_text and int(fixed_box_text) > int(last_detected_speed_sign_text):
                                speed_violations += 1  # Speed violation detected

                    frame_count += 1

                cap.release()
                return speed_violations

            # Process the uploaded video for speed signs
            results = detect_speed_signs(video_path)
            st.write(f"Total Speed Violations: {results}" if results is not None else "No violations detected.")


    # Distance Model Logic
elif st.session_state.page == "Distance Model":
    st.title("Distance Model")

    # Allow users to upload a video with a size limit
    uploaded_file = st.file_uploader("Upload your driving video for distance detection (max size: 100MB)", type=["mp4", "mov", "avi"])

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)  # Convert size to MB
        st.write(f"Uploaded file size: {file_size_mb:.2f} MB")

        if file_size_mb > 100:  # Check if the file size exceeds 100 MB
            st.error("File too large! Please upload a file smaller than 100MB.")
        else:
            # Display the uploaded video
            st.video(uploaded_file)

            # Load the YOLO distance detection model
            model = YOLO('models/DistanceModel.pt')

            # Read the uploaded video using OpenCV
            video_bytes = uploaded_file.read()
            video_path = "/tmp/uploaded_video_distance.mp4"  # Temporarily save the video
            with open(video_path, 'wb') as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Define the dot positions
            center_x = width // 2
            offset = int(0.1 * height)
            offset1 = int(0.25 * height)
            green_dot = (center_x, height - offset1)
            red_dot = (center_x, height - offset)

            # Initialize violation counter and set to keep track of violated cars
            violation_count = 0
            violated_cars = set()
            violation = 0

            # Create a placeholder for the violation count
            violation_placeholder = st.empty()

            # Process the video
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect objects in the frame
                results = model.track(frame, persist=True, show=False)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == 2:  # Class 2 is typically 'car' in YOLO
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            car_bottom = (int((x1 + x2) / 2), y2)
                            car_id = f"{x1}{y1}{x2}_{y2}"
                            if green_dot[1] <= car_bottom[1] < red_dot[1]:
                                if car_id not in violated_cars:
                                    violated_cars.add(car_id)
                            elif car_bottom[1] >= red_dot[1]:
                                violation += 1
                                violation_count = int((violation/100)/2)

                # Update the violation count display
                violation_placeholder.text(f"Processing... Current violations detected: {violation_count}")
            cap.release()

            st.write(f"Total violations detected: {violation_count}")