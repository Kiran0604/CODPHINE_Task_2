import streamlit as st
import tempfile, os, glob, time
from ultralytics import YOLO

#####################
# Video Processing  #
#####################
def process_video(video_path):
    """
    Process the video using YOLOv8 and return a dictionary containing:
      - 'video': path to the processed video (raw output from YOLO)
      - 'images': list of sample image file paths with detection objects
    The YOLO model automatically saves the raw processed output in a subfolder of runs/detect.
    """
    model = YOLO("models/best.pt")
    results = model.predict(source=video_path, show=False, save=True)
    time.sleep(3)
    
    output_folders = glob.glob("runs/detect/*")
    if not output_folders:
        return None
    latest_folder = max(output_folders, key=os.path.getmtime)
    
    video_files = glob.glob(os.path.join(latest_folder, "*.mp4"))
    if not video_files:
        video_files = glob.glob(os.path.join(latest_folder, "*.avi"))
    if not video_files:
        return None
    processed_video = video_files[0]
    
    image_files = glob.glob(os.path.join(latest_folder, "*.jpg"))
    if not image_files:
        image_files = glob.glob(os.path.join(latest_folder, "*.png"))
    image_files.sort()
    sample_images = image_files[:3]
    
    return {"video": processed_video, "images": sample_images}

#####################
# Image Processing  #
#####################
def process_image(image_path):
    """
    Process a single image using YOLOv8 and return:
      - processed_image: path to the processed image
      - detection_info: a dict indicating if a "mask" and/or "helmet" were detected.
    The YOLO model saves the output image in a subfolder of runs/detect.
    """
    model = YOLO("models/best.pt")
    results = model.predict(source=image_path, show=False, save=True)
    time.sleep(2)
    
    output_folders = glob.glob("runs/detect/*")
    if not output_folders:
        return None, None
    latest_folder = max(output_folders, key=os.path.getmtime)
    
    image_files = glob.glob(os.path.join(latest_folder, "*.jpg"))
    if not image_files:
        image_files = glob.glob(os.path.join(latest_folder, "*.png"))
    if not image_files:
        return None, None
    processed_image_path = image_files[0]
    
    # Prepare detection info by checking YOLO results
    detection_info = {"mask": False, "helmet": False}
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        if boxes.cls is not None:
            # Convert detected class tensor to a list of class names
            detected_classes = [results[0].names[int(cls)] for cls in boxes.cls.cpu().numpy()]
            if "mask" in detected_classes:
                detection_info["mask"] = True
            if "helmet" in detected_classes:
                detection_info["helmet"] = True
                
    return processed_image_path, detection_info

#####################
# Sidebar Navigation#
#####################
page = st.sidebar.radio("Navigation", ["Detection (Video)", "Image Detection", "Safety Measures Blog"])

#####################
# Video Detection   #
#####################
if page == "Detection (Video)":
    st.title("PPE Detection on Construction Sites - Video")
    st.write("Upload a video file (mp4 or avi) to detect PPE using the YOLO model.")
    
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        st.subheader("Original Uploaded Video")
        st.video(tfile.name)
        
        if st.session_state.processed_data is None:
            with st.spinner("Processing video, please wait..."):
                st.session_state.processed_data = process_video(tfile.name)
        
        if st.session_state.processed_data is not None:
            st.success("Video processed successfully!")
            
            with open(st.session_state.processed_data["video"], "rb") as f:
                processed_video_bytes = f.read()
            st.download_button(
                label="Download Processed Video",
                data=processed_video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
            
            if st.session_state.processed_data["images"]:
                st.subheader("Sample Processed Images")
                for img_file in st.session_state.processed_data["images"]:
                    st.image(img_file, caption=os.path.basename(img_file), use_container_width=True)
            else:
                st.warning("No processed images found to display.")
            
            st.subheader("Recommendations")
            st.markdown("""
            - **Lighting & Angle:** Ensure the camera angle and lighting provide a clear view of the personnel.
            - **Model Calibration:** Adjust the detection confidence threshold if you experience too many false positives/negatives.
            - **Training Data:** Consider including more varied examples of PPE in different environments to improve detection accuracy.
            - **Regular Updates:** Update the model periodically with new data to adapt to changing conditions on the site.
            """)
        else:
            st.error("Error: Could not locate the processed video. Please try again.")

#####################
# Image Detection   #
#####################
elif page == "Image Detection":
    st.title("🦺 PPE Detection System - Image")
    st.write("Upload an image file (jpg or png) to detect PPE using the YOLO model.")
    
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "png"])
    
    if uploaded_image is not None:
        img_temp = tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_image.name.split('.')[-1])
        img_temp.write(uploaded_image.read())
        img_temp.flush()
        
        st.markdown("<h4>📸 Uploaded Image:</h4>", unsafe_allow_html=True)
        st.image(img_temp.name, use_container_width=True)
        
        with st.spinner("Processing image, please wait..."):
            processed_img, detection_info = process_image(img_temp.name)
        
        if processed_img:
            st.markdown("<h4>🔍 Detected Results:</h4>", unsafe_allow_html=True)
            st.image(processed_img, use_container_width=True)
            
            # Check for missing PPE (mask and helmet)
            missing = []
            if not detection_info["mask"]:
                missing.append("mask")
            if not detection_info["Hardhat"]:
                missing.append("Hardhat")
                
            if missing:
                st.error("Warning: No " + " and ".join(missing) + " detected!")
            else:
                st.success("All required PPE detected!")
        else:
            st.error("Error: Could not process the image. Please try again.")

##############################
# Safety Measures Blog Page  #
##############################
elif page == "Safety Measures Blog":
    st.title("Safety Measures Blog")
    st.markdown("""
    ### Construction Site Safety: Key Measures to Protect Your Workforce
    
    In the dynamic environment of a construction site, safety should always be the top priority. Here are some essential measures:
    
    1. **Personal Protective Equipment (PPE):**  
       Always ensure that all personnel are equipped with the necessary PPE such as helmets, gloves, high-visibility vests, and protective footwear.
    
    2. **Site Inspections and Hazard Identification:**  
       Regularly conduct thorough site inspections to identify potential hazards. Implement corrective measures promptly.
    
    3. **Training and Awareness:**  
       Provide regular training sessions on safety practices and emergency procedures. Educate workers about the importance of PPE and how to use it correctly.
    
    4. **Emergency Preparedness:**  
       Establish clear protocols for emergency situations. Ensure that emergency exits, first aid kits, and communication devices are easily accessible.
    
    5. **Regular Maintenance:**  
       Keep all equipment and machinery in good working order with routine maintenance checks to avoid malfunctions that could lead to accidents.
    
    By integrating these measures, construction sites can minimize risks and protect workers, ensuring a safer work environment for everyone involved.
    """)
