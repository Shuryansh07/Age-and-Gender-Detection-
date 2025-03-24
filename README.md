# 🎭 Age & Gender Detection  

An AI-powered **Age & Gender Detection** system using OpenCV and deep learning. This project leverages pre-trained **Caffe models** to detect faces and classify age and gender in real-time. 🚀  

## 📝 Project Overview  
This Python-based application detects a person's **age range** and **gender** from a webcam feed. It uses **OpenCV's deep learning module (cv2.dnn)** to load pre-trained models for face detection and classification.  

## 📌 Features  
✅ **Real-time Face Detection** using OpenCV  
✅ **Predicts Age & Gender** from a live webcam feed  
✅ **Pre-trained Caffe Models** for fast and accurate classification  
✅ **User-friendly** – Just run the script, and it works!  
✅ **Stop the program anytime** by pressing **'q'**  

---

## 🛠 Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Shuryansh07/Age-Gender-Detection-.git
cd Age-Gender-Detection
```

### 2️⃣ Install Dependencies  
Make sure you have Python installed. Then, install the required libraries:  
```bash
pip install opencv-python numpy
```

---

## 🚀 Usage  

Simply run the **main script**, and the program will start detecting age and gender automatically!  

```bash
python main.py
```

🔹 Press **'q'** to exit the program.  

---

## 📜 Code Explanation  

```python
import cv2

# Function to detect faces
def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [184,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxes

# Load model files
faceProto = "deploy_age.prototxt"
faceModel = "age_net.caffemodel"
ageProto = "deploy_age.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "deploy_gender.prototxt"
genderModel = "gender_net.caffemodel"

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define model mean values and labels
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Start webcam capture
video = cv2.VideoCapture(0)
padding = 20

while True:
    ret, frame = video.read()
    frame, bboxes = faceBox(faceNet, frame)
    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        # Predict age
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        label = "{}, {}".format(gender, age)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Age-Gender", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
```

---

## 📂 Project Structure  

```
Age-Gender-Detection/
│── age_net.caffemodel
│── gender_net.caffemodel
│── deploy_age.prototxt
│── deploy_gender.prototxt
│── main.py
│── README.md
```

---

## 📌 How It Works  
1️⃣ The program captures frames from a **webcam**.  
2️⃣ A **deep learning model** detects faces in the frame.  
3️⃣ Each detected face is passed through a **Caffe model** to predict:  
   - **Gender** (Male/Female)  
   - **Age range** (e.g., 0-2, 4-6, etc.)  
4️⃣ The predicted age and gender are displayed on the screen.  
5️⃣ Press **'q'** to quit the program.  

---

## 🤖 Pre-trained Models  
This project uses pre-trained **Caffe models** trained on the **Adience dataset** for age and gender classification.  

---

## ⚡ Future Improvements  
- 🔹 Improve model accuracy with a custom-trained dataset  
- 🔹 Deploy as a **web app** using Flask/Streamlit  
- 🔹 Optimize performance for **faster inference** 

---

## 📝 License  
This project is **open-source** and available under the **MIT License**.  

📢 **Contributions are welcome!** Feel free to open issues or submit pull requests.  

---

🚀 **Happy Coding!** 🎯  
