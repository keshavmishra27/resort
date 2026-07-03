# Eco Resort => Waste Classification System

An intelligent web application that uses Machine Learning to classify waste images and an interactive scoring system to gamify environmental sustainability.

##  Overview

The application allows users to upload images of waste. The backend utilizes a deep learning model to predict the class of the waste (e.g., Biodegradable, E waste, Hazardous) and uses computer vision techniques to detect and count the number of objects. Based on the classification and count, users are awarded points, fostering an engaging leaderboard experience.

---

##  System Architecture

The project is built on a modular architecture separating the web server, database, and machine learning components.

```mermaid
graph TD
    A[Client Browser] -->|HTTP POST Image| B(Flask Web Server)
    B --> C{Authentication}
    C -->|Valid User| D(Image Processing Pipeline)
    
    subgraph Machine Learning Pipeline
    D --> E[MobileNetV2 Classifier]
    D --> F[OpenCV Object Counter]
    E --> G[Waste Category Prediction]
    F --> H[Object Count]
    end
    
    G --> I[Scoring Engine]
    H --> I
    I --> J[(SQLite Database)]
    J --> K[Leaderboard Update]
    K --> A
```

---

##  User Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant App as Web App (Flask)
    participant ML as ML Pipeline
    participant DB as Database

    User->>App: Register / Login
    App->>DB: Validate Credentials
    DB-->>App: Access Granted
    User->>App: Upload Image
    App->>ML: Pass Image for Inference
    ML->>ML: Classify Waste Type
    ML->>ML: Count Objects via CV2
    ML-->>App: Return Class & Count
    App->>App: Calculate Score (Class * Count)
    App->>DB: Add Score to User Profile
    App-->>User: Display Results & Annotated Image
```

---

##  Scoring System Table

Each waste classification corresponds to a specific base score. The total score for an upload is calculated as `Base Score × Number of Objects Detected`.

| Waste Classification                     | Base Score per Item | Impact Level  |
| ---------------------------------------- | :-----------------: | ------------- |
| **Biodegradable**                        | 10                  | Low           |
| **Ewaste**                               | 20                  | Medium        |
| **Hazardous**                            | 30                  | High          |
| **Non Biodegradable**                    | 40                  | Very High     |
| **Pharmaceutical and Biomedical Waste**  | 50                  | Critical      |

---

##  Technology Stack

- **Backend:** Flask, Flask Login, Flask WTF
- **Machine Learning:** TensorFlow / Keras (MobileNetV2 base)
- **Computer Vision:** OpenCV
- **Database:** SQLAlchemy (SQLite)
- **Security:** Bcrypt (Password Hashing)

---

##  Setup & Installation

**1. Clone the repository and navigate to the project directory**
```bash
cd resort
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Application**
```bash
python run.py
```

The server will start on `http://127.0.0.1:5000/`. You can navigate here in your browser, register an account, and start uploading images!

---

##  Model Training & Testing

**1. Training the Model**
If you wish to retrain the image classification model, you can run the `train_model.py` script. The script handles data loading, model compilation, training, and testing metrics evaluation. It will save the resulting model to `backend/models/garbage_tf_model.h5`, and generate loss/accuracy plots and a confusion matrix.
```bash
python backend\train_model.py
```

**2. Testing on Uploaded Images**
To quickly test the trained model on raw, unlabeled images, place your `.jpg` or `.png` files in the `backend\static\uploads` folder. Then, run the testing script to output the predicted class and confidence score for each image in the terminal:
```bash
python backend\test.py
```

**3. Environment Variables (Optional)**
The training and testing scripts use `python-dotenv` to manage file paths securely. If you want to use custom absolute paths on your computer without exposing them in GitHub, create a `.env` file in the root directory:
```env
UPLOADS_DIR="absolute\path\to\resort\backend\static\uploads"
MODEL_PATH="absolute\path\to\resort\backend\models\garbage_tf_model.h5"
TRAIN_DIR="absolute\path\to\resort\dataset\splitted\train"
VAL_DIR="absolute\path\to\resort\dataset\splitted\valid"
TEST_DIR="absolute\path\to\resort\dataset\splitted\test"
```
If you don't create a `.env` file, the scripts will automatically fall back to safe relative paths inside the project folder.

**4. Version Control**
A `.gitignore` file is properly configured to ignore large auto-generated files (datasets, model weights `*.h5`, cache, generated plots, your `venv`, and the `.env` file), keeping your repository clean and your local absolute paths private!