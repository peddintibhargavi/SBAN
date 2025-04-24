from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torch
import os
import sys
import uuid
import pickle
from typing import Dict, List, Optional
import uvicorn
from pydantic import BaseModel

# Add the directory containing the modules to the Python path
MODULE_DIR = os.path.dirname(__file__)
sys.path.insert(0, MODULE_DIR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../FaceUnimodel'))
from feature_conv import float_to_q1_8, q1_8_to_float, secure_enrollment, secure_decrypt, generate_octets
from dot_comapre import distribute_template_shares, secure_and_masked_dot_product, compute_hamming_distance_bits, apply_correction
from detect_noise import resnet_50, Tr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../common'))
from one_parameter_defense import one_parameter_defense

app = FastAPI(title="Face Biometric Authentication API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory database of users (in production, use a real database)
user_database: Dict[str, Dict] = {}
# Mapping of session tokens to user IDs
active_sessions: Dict[str, str] = {}

# Define data models
class UserCreate(BaseModel):
    username: str
    full_name: str

class User(BaseModel):
    user_id: str
    username: str
    full_name: str

class EnrollmentResponse(BaseModel):
    success: bool
    message: str
    user_id: str

class AuthenticationResponse(BaseModel):
    authenticated: bool
    session_token: Optional[str] = None
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    message: str

class Enrollment(BaseModel):
    user_id: str
    embedding: List[float]

# Define storage paths
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
EMBEDDING_DIR = os.path.join(os.path.dirname(__file__), "embeddings")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

def save_embedding(user_id: str, embedding: np.ndarray, R1: np.ndarray, M: np.ndarray):
    """Save user enrollment data to disk with encryption keys (called ONLY during enrollment)"""
    file_path = os.path.join(EMBEDDING_DIR, f"{user_id}.pkl")
    if os.path.exists(file_path):
        raise RuntimeError(f"Embedding already exists for user {user_id}. Aborting overwrite.")
    
    data = {
        "embedding": embedding,
        "R1": R1,
        "M": M
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_embedding(user_id: str):
    """Load user enrollment data including encryption keys"""
    file_path = os.path.join(EMBEDDING_DIR, f"{user_id}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No enrollment data found for user {user_id}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)
def process_image(image_path: str) -> np.ndarray:
    """
    Extracts identity vector from the full face image using resnet_50.
    This replaces the incorrect use of noise vector for authentication.
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (224, 224))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Convert to tensor
    tensor = Tr(Image.fromarray(rgb_image)).unsqueeze(0)  # shape: [1, 3, 224, 224]

    # Get feature from ResNet50
    with torch.no_grad():
        embedding = resnet_50(tensor).squeeze().numpy()

    # Normalize the embedding
    return embedding / np.linalg.norm(embedding)

def verify_similarity(enrolled_data, verification_embedding: np.ndarray) -> tuple:
    """
    Verifies similarity between enrolled and verification embeddings using stored keys.
    Returns (is_authenticated, score)
    """
    # Extract enrolled data
    enrolled_embedding = enrolled_data["embedding"]
    R1 = enrolled_data["R1"]
    M = enrolled_data["M"]
    
    # Confirm dimensions match
    if enrolled_embedding.shape[0] != verification_embedding.shape[0]:
        raise ValueError(f"Embedding dimension mismatch: {enrolled_embedding.shape[0]} vs {verification_embedding.shape[0]}")
    
    # Get embedding size
    embedding_size = enrolled_embedding.shape[0]
    
    # Convert to Q1.8 format
    X = float_to_q1_8(enrolled_embedding)  # Already stored in enrollment
    Y = float_to_q1_8(verification_embedding)
    N = float_to_q1_8(np.random.uniform(-1, 1, size=embedding_size) * 50)  # Random mask for Y
    
    # Encrypt verification embedding - THIS IS CRUCIAL: Use a NEW R2 for Y!
    Y_enc, N_enc, R2 = secure_enrollment(Y, N)
    
    # Re-encrypt X with STORED R1 from enrollment
    X_enc = np.bitwise_xor(X, R1)
    M_enc = np.bitwise_xor(M, R1)
    
    # Distribute shares
    P1, P2 = distribute_template_shares(X_enc, Y_enc, M_enc, N_enc)

    # Process a consistent number of bits
    # Use multiple of 16 to ensure proper octet generation
    num_bits = 128  # Choose a power of 2 that divides evenly by 4 for octets
    num_elements = num_bits // 16  # 16 bits per element in int16
    
    # Convert encrypted templates to binary and generate octets
    def int_to_bits(arr, bit_width=16):
        # Ensure proper bit ordering and consistent shape
        bits = np.unpackbits(arr.view(np.uint8))
        return bits.reshape(-1, bit_width)[:, ::-1]  # MSB last

    # Ensure we have enough elements and use fixed size
    if P1['X'].shape[0] < num_elements or P2['Y'].shape[0] < num_elements:
        raise ValueError(f"Not enough elements in templates: X={P1['X'].shape[0]}, Y={P2['Y'].shape[0]}, need {num_elements}")
    
    # Get exactly num_elements worth of bits
    octets_X = generate_octets(int_to_bits(P1['X'][:num_elements]).flatten()[:num_bits])
    octets_Y = generate_octets(int_to_bits(P2['Y'][:num_elements]).flatten()[:num_bits])
    octets_M = generate_octets(int_to_bits(P1['M'][:num_elements]).flatten()[:num_bits])
    octets_N = generate_octets(int_to_bits(P2['N'][:num_elements]).flatten()[:num_bits])

    # Ensure all octets have same shape
    assert octets_X.shape == octets_Y.shape == octets_M.shape == octets_N.shape, \
           f"Octet shape mismatch: X={octets_X.shape}, Y={octets_Y.shape}, M={octets_M.shape}, N={octets_N.shape}"

    # Secure masked AND dot product
    raw_score = secure_and_masked_dot_product(P1, P2, octets_X, octets_Y, octets_M, octets_N)

    # Compute observed Hamming distance between octets_X and octets_Y
    avg_bits_diff, observed_hd = compute_hamming_distance_bits(octets_X, octets_Y)

    # Apply correction if observed HD deviates from expected 25%
    corrected_score = apply_correction(raw_score, observed_hd, expected_hd=0.25)

    # Log actual scores for debugging
    print(f"Raw score: {raw_score}, HD: {observed_hd:.4f}, Corrected: {corrected_score:.4f}")

    # Authentication decision
    # Adjust these thresholds based on your system testing
    max_threshold = 28
    min_threshold = 3
    is_authenticated = corrected_score >= min_threshold and corrected_score <= max_threshold
    is_authenticated, corrected_score = secure_comparison(enrolled_data, verification_embedding)
    
    # If not authenticated, fall back to cosine similarity
    if not is_authenticated:
        cosine_sim = np.dot(enrolled_data["embedding"], verification_embedding)
        is_authenticated = cosine_sim > 0.85  # Adjust threshold based on testing
        
    return is_authenticated, corrected_score
def cleanup_temp_files(file_path: str):
    """Delete temporary files after processing"""
    if os.path.exists(file_path):
        os.remove(file_path)

@app.post("/users/", response_model=User)
async def create_user(user: UserCreate):
    """Create a new user in the system"""
    user_id = str(uuid.uuid4())
    user_data = {
        "user_id": user_id,
        "username": user.username,
        "full_name": user.full_name,
        "enrolled": False
    }
    user_database[user_id] = user_data
    return user_data

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    """Get user information"""
    if user_id not in user_database:
        raise HTTPException(status_code=404, detail="User not found")
    return user_database[user_id]

@app.post("/enroll/", response_model=EnrollmentResponse)
async def enroll_user(
    user_id: str, 
    background_tasks: BackgroundTasks, 
    face_image: UploadFile = File(...), 
    full_name: str = Form(None)
):
    """Enroll a user with their facial biometric"""
    # Check if user exists
    if user_id not in user_database:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update full name if provided
    if full_name:
        user_database[user_id]["full_name"] = full_name
    
    # Save uploaded image
    image_path = os.path.join(UPLOAD_DIR, f"{user_id}_enrollment_{uuid.uuid4()}.jpg")
    with open(image_path, "wb") as buffer:
        buffer.write(await face_image.read())
    
    try:
        # Process image to extract embedding
        embedding = process_image(image_path)
        
        # Convert to Q1.8 and generate masks for secure enrollment
        X = float_to_q1_8(embedding)
        M = float_to_q1_8(np.random.uniform(-1, 1, size=embedding.shape[0]) * 50)
        
        # Secure enrollment (encrypt templates)
        X_enc, M_enc, R1 = secure_enrollment(X, M)
        
        # Save enrollment data including keys
        save_embedding(user_id, embedding, R1, M)
        
        # Update user status
        user_database[user_id]["enrolled"] = True
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, image_path)
        
        return EnrollmentResponse(
            success=True,
            message="User enrolled successfully",
            user_id=user_id
        )
    
    except Exception as e:
        # Schedule cleanup even on error
        background_tasks.add_task(cleanup_temp_files, image_path)
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")
@app.post("/authenticate/", response_model=AuthenticationResponse)
async def authenticate_user(background_tasks: BackgroundTasks, face_image: UploadFile = File(...)):
    """
    Authenticate a user using facial biometric.
    The function will try to match the provided face against all enrolled users.
    """
    # Save uploaded image
    image_path = os.path.join(UPLOAD_DIR, f"auth_{uuid.uuid4()}.jpg")
    with open(image_path, "wb") as buffer:
        buffer.write(await face_image.read())
    
    try:
        # Process image to extract verification embedding
        verification_embedding = process_image(image_path)
        
        # Try to match against all enrolled users
        best_match = None
        best_score = 0
        user_name = None
        
        for user_id, user_data in user_database.items():
            if not user_data.get("enrolled", False):
                continue
                
            try:
                # Load user's enrolled data (including R vectors)
                enrolled_data = load_embedding(user_id)
                
                # Compare embeddings
                is_authenticated, score = verify_similarity(enrolled_data, verification_embedding)
                
                if is_authenticated and (best_match is None or score > best_score):
                    best_match = user_id
                    best_score = score
                    user_name = user_data.get("full_name", "Unknown")
                    
            except FileNotFoundError:
                # Skip users with missing embeddings
                continue
            except Exception as e:
                print(f"Error verifying user {user_id}: {str(e)}")
                continue
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, image_path)
        
        if best_match:
            # Generate session token for authenticated user
            session_token = str(uuid.uuid4())
            active_sessions[session_token] = best_match
            
            return AuthenticationResponse(
                authenticated=True,
                session_token=session_token,
                user_id=best_match,
                user_name=user_name,
                message=f"Authentication successful with score: {best_score:.2f}"
            )
        else:
            return AuthenticationResponse(
                authenticated=False,
                session_token=None,
                user_id=None,
                user_name=None,
                message="Authentication failed: No matching user found"
            )
    
    except Exception as e:
        # Schedule cleanup even on error
        background_tasks.add_task(cleanup_temp_files, image_path)
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
@app.post("/logout/")
async def logout(session_token: str):
    """Logout a user by invalidating their session token"""
    if session_token in active_sessions:
        del active_sessions[session_token]
        return {"success": True, "message": "Logged out successfully"}
    else:
        raise HTTPException(status_code=400, detail="Invalid session token")

@app.get("/healthcheck")
async def healthcheck():
    """API health check endpoint"""
    return {"status": "online", "service": "Face Biometric Authentication API"}
@app.post("/debug_enrollment/{user_id}")
async def debug_enrollment(user_id: str):
    """Debug endpoint to check enrollment data"""
    if user_id not in user_database or not user_database[user_id].get("enrolled", False):
        raise HTTPException(status_code=404, detail="User not enrolled")
    
    try:
        # Load enrollment data
        enrolled_data = load_embedding(user_id)
        
        # Return summary of data shapes for debugging
        return {
            "embedding_shape": enrolled_data["embedding"].shape,
            "R1_shape": enrolled_data["R1"].shape,
            "M_shape": enrolled_data["M"].shape,
            "embedding_stats": {
                "mean": float(np.mean(enrolled_data["embedding"])),
                "std": float(np.std(enrolled_data["embedding"])),
                "min": float(np.min(enrolled_data["embedding"])),
                "max": float(np.max(enrolled_data["embedding"]))
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading enrollment data: {str(e)}")
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)