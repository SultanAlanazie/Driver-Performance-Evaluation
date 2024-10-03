import streamlit as st
import cv2
import csv
import numpy as np
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain_groq import ChatGroq
from ultralytics import YOLO

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

# Define prompt template for the chatbot
prompt = PromptTemplate(
    input_types={'subject': 'string', 'language': 'string'},
    template="""You are a driving assistant bot. do not get out of the topic about driving and answer the subject: {subject}, that the user provided if it's related to the driving, **your answers will be in arabic**"""
)

# Initialize the chain for the chatbot
chain = LLMChain(llm=llm, prompt=prompt)

# Sidebar for navigation
st.sidebar.title("Navigation")
# Initialize a session state variable for page tracking
if 'page' not in st.session_state:
    st.session_state.page = "Welcome"

# Create buttons for each page
if st.sidebar.button("ChatBot"):
    st.session_state.page = "Welcome"
if st.sidebar.button("Stop Sign Model"):
    st.session_state.page = "Stop Sign Model"
if st.sidebar.button("Speed Model"):
    st.session_state.page = "Speed Model"
if st.sidebar.button("Distance Model"):
    st.session_state.page = "Distance Model"

# Welcome Page with Chatbot
if st.session_state.page == "Welcome":
    st.title("ChatBot")
    st.write("مرحبا")

    # Display the welcome message
    welcome_message = """! كيف يمكنني مساعدتك؟ من فضلك اختر أحد الخيارات التالية:\n
    1. شرح حول كيفية استخدام النظام\n
    2. متطلبات الفيديو\n
    3. فوائد النظام\n"""
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

    # Allow users to upload a video with a size limit
    uploaded_file = st.file_uploader("Upload your driving video for speed detection (max size: 100MB)", type=["mp4", "mov", "avi"])

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)  # Convert size to MB
        st.write(f"Uploaded file size: {file_size_mb:.2f} MB")

        if file_size_mb > 100:  # Check if the file size exceeds 100 MB
            st.error("File too large! Please upload a file smaller than 100MB.")
        else:
            # Display the uploaded video
            st.video(uploaded_file)

            # Load the speed detection model
            model = YOLO('models/SpeedModel.pt')

            # Read the uploaded video using OpenCV
            video_bytes = uploaded_file.read()
            video_path = "/tmp/uploaded_video_speed.mp4"  # Temporarily save the video
            with open(video_path, 'wb') as f:
                f.write(video_bytes)
            def extract_region(image, box):
                x1, y1, x2, y2 = map(int, box)
            # Function to detect speed signs and violations
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

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Object detection every frame_interval frames
                    if frame_count % frame_interval == 0:
                        results = model.track(frame, persist=True, show=False)

                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                conf = box.conf[0]
                                cls = int(box.cls[0])

                                # Check for speed sign detection (class 0)
                                if conf > confidence_threshold and cls == 0:
                                    speed_sign_region = extract_region(frame, box.xyxy[0].cpu().numpy())
                                    speed_sign_texts = read_text_with_easyocr(speed_sign_region)

                                    # Extract digit text from the detected speed sign
                                    speed_sign_text = ''.join([text for text, prob in speed_sign_texts if text.isdigit()])

                                    if speed_sign_text:
                                        last_detected_speed_sign_text = speed_sign_text
                                        st.write(f"Speed sign detected: {speed_sign_text} km/h at frame {frame_count}")

                    # Perform OCR on fixed box every ocr_interval frames
                    if frame_count % ocr_interval == 0 and last_detected_speed_sign_text:
                        fixed_box = (850, 950, 50, 90)  # Adjust the position accordingly
                        fixed_box_region = extract_region(frame, fixed_box)
                        fixed_box_texts = read_text_with_easyocr(fixed_box_region)

                        # Extract digit text from fixed box
                        fixed_box_text = ''.join([text for text, prob in fixed_box_texts if text.isdigit()])

                        if fixed_box_text:
                            if int(fixed_box_text) > int(last_detected_speed_sign_text):
                                speed_violations += 1

                    frame_count += 1

                cap.release()
                return speed_violations

            # Process the uploaded video and count speed violations
            violations = detect_speed_signs(video_path)
            st.write(f"Total Speed Violations: {violations}")

# Distance Model Logic

# Streamlit code to handle "Distance Model" page
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