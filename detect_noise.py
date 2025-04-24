import cv2
import numpy as np
import sys
import mediapipe as mp
from PIL import Image
import itertools
from hair_segmentation import *
import torch
# from torchsr.models import ninasr_b0
import torch.nn as nn
import os
from torchvision import transforms
import torchvision.models as models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../common'))
from one_parameter_defense import one_parameter_defense

class NoiseExtractorCNN(nn.Module):
    """3-layer CNN for noise feature extraction"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*14*14, 128)  # For 64x64 input
        )
    
    def forward(self, x):
        return self.layers(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
weights = models.ResNet50_Weights.IMAGENET1K_V1
# preprocess = weights.transforms()
resnet_50 = models.resnet50(weights=weights)
# print(mobilenet_v3_small)
# mobilenet_v3_rep = torch.nn.Sequential(*(list(mobilenet_v3_small.children())[:-1]))
# until_last_layer = torch.nn.Sequential(*(list(mobilenet_v3_small.classifier.children())[:-1]))
# mobilenet_v3_rep = torch.nn.Sequential(mobilenet_v3_rep, torch.nn.Flatten(),  until_last_layer).eval()
resnet_50 =  torch.nn.Sequential(*(list(resnet_50.children())[:-1])).eval()


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
transform = transforms.Compose([transforms.ToTensor()])
# sr_model = ninasr_b0(scale=4, pretrained=True).eval()
# sr = cv2.dnn_superres.DnnSuperResImpl_create()
# sr.readModel('../dumps/EDSR_x4.pb')
# sr.setModel("edsr",4)
hairsegmentation = HairSegmentation(1024, 1024)

Tr = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

path = None
if len(sys.argv) > 1:
    path = sys.argv[1]

# detecting ears
# basic tree based volia-jones, doesn't seem to work for non-frontal images
def detect_haarcascade(path):
    left_ear_cascade = cv2.CascadeClassifier('./cascades/haarcascade_mcs_leftear.xml')
    right_ear_cascade = cv2.CascadeClassifier('./cascades/haarcascade_mcs_rightear.xml')
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface.xml')
    lefteye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_lefteye.xml')
    righteye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_righteye.xml')
    # eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')

    img = cv2.imread(path)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left_ear = left_ear_cascade.detectMultiScale(gray, 1.3, 1)
    right_ear = right_ear_cascade.detectMultiScale(gray, 1.3, 1)
    face = face_cascade.detectMultiScale(gray, 1.3 , 5)
    left_eye = lefteye_cascade.detectMultiScale(gray, 1.5 , 5)
    right_eye = righteye_cascade.detectMultiScale(gray, 1.5 , 5)

    for (x,y,w,h) in left_ear:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

    for (x,y,w,h) in right_ear:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)
    
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)
    
    for (x,y,w,h) in left_eye:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)


    for (x,y,w,h) in right_eye:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)

    cv2.imwrite('example_har.png', img)

def detect_mediapipe(img_or_path):
    """
    Extract facial regions using MediaPipe
    
    Args:
        img_or_path: Either an image array or a path to an image file
        
    Returns:
        tuple: (left_eye_region, right_eye_region, hair_region) - regions of interest
    """
    FACE_OVAL = list(set(itertools.chain(*mp_face_mesh.FACEMESH_FACE_OVAL)))
    # Left eye indices list
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    # Right eye indices list
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6) as face_mesh:
        # Check if input is a path or an image
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path
            
        img = cv2.resize(img, dsize=(1024, 1024))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            img_h, img_w = 1024, 1024
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            # create mask for hair
            masked_img_hair = hairsegmentation(rgb)
            masked_img_hair = np.stack([masked_img_hair, masked_img_hair, masked_img_hair], axis=-1)
            masked_img = np.zeros_like(img)
            mp_drawing.draw_landmarks(
                    image=masked_img,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            th, im_th = cv2.threshold(masked_img_gray, 200, 255, cv2.THRESH_BINARY_INV)
            im_floodfill = im_th.copy()
            h, w = im_th.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(im_floodfill, mask, (0,0), (0,0,0))
            masked_img_hair[im_floodfill > 0] = 0
            
            # create mask for the eyes
            extracted_img = extract_masked_regions(img, masked_img_hair) #hair
            x,y,w,h = cv2.boundingRect(mesh_points[LEFT_EYE])
            left_eye = img[y : y+h, x : x+w, :]
            cv2.imwrite('example_mediapipe_lefteye.png', img[y : y+h, x : x+w, :])
            x,y,w,h = cv2.boundingRect(mesh_points[RIGHT_EYE])
            right_eye = img[y : y+h, x : x+w, :]
            cv2.imwrite('example_mediapipe_righteye.png', img[y : y+h, x : x+w, :])
            cv2.imwrite('example_mediapipe_hair.png', extracted_img)
            cv2.fillPoly(masked_img_hair, [mesh_points[LEFT_EYE]], (255,255,255))
            cv2.fillPoly(masked_img_hair, [mesh_points[RIGHT_EYE]], (255,255,255))
            extracted_img_1 = extract_masked_regions(img, masked_img_hair) # eye and hair
            cv2.imwrite('mediapipe_hair_eye.png', extracted_img_1)
            extracted_img_2 = extract_masked_regions_overlay(img, masked_img_hair) # overlay on image
            cv2.imwrite('mediapipe_hair_eye_overlay.png', extracted_img_2)
            
            # Return the three regions needed by process_image
            return left_eye, right_eye, extracted_img
        else:
            print("No face detected in the image.")
            return None, None, None
    # Convert regions to feature vectors
    def _region_to_vector(region):
        region = cv2.resize(region, (128, 128))
        tensor = transform(Image.fromarray(region)).unsqueeze(0)
        with torch.no_grad():
            return resnet_50(tensor).flatten().numpy()

    noise_vec = np.concatenate([
        _region_to_vector(left_eye),
        _region_to_vector(right_eye),
        _region_to_vector(hair_region)
    ])

    # Enforce noise-identity decorrelation
    corr = np.corrcoef(identity_vector, noise_vec)[0, 1]
    if abs(corr) > 0.15:
        noise_vec = one_parameter_defense(noise_vec, m=10)  # Strong perturbation

    return noise_vec
def extract_noise(image_path: str, identity_vector: np.ndarray) -> np.ndarray:
    """Enhanced noise extraction with CNN"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1024, 1024))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize models
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    hair_seg = HairSegmentation(1024, 1024)
    noise_cnn = NoiseExtractorCNN().eval()
    
    # Process image
    results = face_mesh.process(rgb)
    landmarks = results.multi_face_landmarks[0].landmark
    mesh_points = np.array([[int(l.x * 1024), int(l.y * 1024)] for l in landmarks])
    
    # Extract regions
    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155]
    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249]
    
    left_eye = extract_region(img, mesh_points[left_eye_indices])
    right_eye = extract_region(img, mesh_points[right_eye_indices])
    hair_mask = hair_seg(rgb)
    hair_region = extract_masked_regions(rgb, hair_mask)
    
    # Extract noise features
    with torch.no_grad():
        def process_region(region):
            region = cv2.resize(region, (64, 64))
            return noise_cnn(transform(region).unsqueeze(0))
        
        noise_vec = torch.cat([
            process_region(left_eye),
            process_region(right_eye),
            process_region(hair_region)
        ], dim=1).squeeze().numpy()
    
    # Ensure decorrelation
    corr = np.abs(np.dot(identity_vector, noise_vec))
    if corr > 0.15:
        noise_vec = one_parameter_defense(noise_vec, m=10)
    
    return noise_vec
def extract_region(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Extract bounded region from image."""
    x, y, w, h = cv2.boundingRect(points)
    return img[y:y+h, x:x+w]
def extract_identity_vector_from_mediapipe(image: np.ndarray) -> np.ndarray:
    import cv2
    from PIL import Image
    from detect_noise import resnet_50, Tr

    resized = cv2.resize(image, (224, 224))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = Tr(Image.fromarray(rgb_image)).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet_50(tensor).squeeze().numpy()
        return embedding / np.linalg.norm(embedding)
def extract_masked_regions(img, mask):
    """Extract regions from image based on mask"""
    result = np.zeros_like(img)
    result[mask > 0] = img[mask > 0]
    return result
def extract_masked_regions_overlay(img, mask):
    """
    Overlay the mask on the original image with a specific color.
    Args:
        img: Original image (numpy array).
        mask: Binary mask (numpy array).
    Returns:
        Overlayed image.
    """
    # Ensure the overlay is a 3-channel image
    overlay = img.copy()
    
    # Make sure mask is binary and has proper dimensions
    if mask.ndim == 3 and mask.shape[2] == 3:
        # If mask is already 3D, take just one channel for the binary mask
        mask_binary = (mask[:, :, 0] > 0).astype(np.uint8)
    else:
        # Otherwise use the mask as is
        mask_binary = (mask > 0).astype(np.uint8)
    
    # Apply the green color to the masked regions
    for i in range(3):
        # Only change the green channel to 255, leave others at 0
        if i == 1:  # Green channel
            overlay[:, :, i] = np.where(mask_binary > 0, 255, overlay[:, :, i])
        else:  # Red and Blue channels
            overlay[:, :, i] = np.where(mask_binary > 0, 0, overlay[:, :, i])
    
    return overlay
if __name__ == "__main__":
    if path:
        detect_haarcascade(path)
        detect_mediapipe(path)
