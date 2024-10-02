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
            model = YOLO("/workspaces/Driver-Performance-Evaluation/models/StopModel.pt")  # Use YOLO directly

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
                        boxes = result.boxes  # Access the 'boxes' attribute

                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()[:4].astype(int)
                            conf, class_id = box.conf[0], box.cls[0]

                            if int(class_id) == STOP_SIGN_CLASS_ID:
                                stop_sign_in_frame = True
                                stop_sign_detected_frame_count += 1

                                if not stop_sign_detected and stop_sign_detected_frame_count >= REQUIRED_STOP_FRAMES:
                                    stop_sign_id += 1
                                    stop_sign_detected = True
                                    sign_logged = False
                                    total_stop_signs += 1  # Increment total stop signs encountered

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"Stop Sign: {conf:.2f}"
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
                        writer.writerow({'stop_sign_id': stop_sign_id, 'stopped': stop_completed})
                        stop_sign_detected = False
                        stop_completed = False
                        stopped_frame_count = 0
                        stop_sign_detected_frame_count = 0
                        # Increment stopped or not stopped counter
                        if stop_completed:
                            total_stopped += 1
                        else:
                            total_not_stopped += 1

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

elif st.session_state.page == "Speed Model":
    st.title("Speed Model")
    # Write Speed logic here

elif st.session_state.page == "Distance Model":
    st.title("Distance Model")
    # Write Distance logic here


