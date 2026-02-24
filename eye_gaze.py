import os
import sys
import numpy as np
import torch
import yaml
from pathlib import Path
import cv2


EYE_TRACKING_PATH = "./"
sys.path.insert(0, EYE_TRACKING_PATH)

from inference.model import backbone
from inference.model.head import SocialEyePredictionBoundHead
from inference.model.model import SocialEyeModel
from inference.model.model_utils import load_checkpoint


class EyeGazeInferenceBatch:
    """
    Batch inference for eye gaze estimation.
    """
    
    def __init__(
        self, 
        model_checkpoint_path=None,
        model_config_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the eye gaze inference model.
        
        Args:
            model_checkpoint_path: Path to model weights
            model_config_path: Path to model config
            device: Device to run inference on
        """
        # Default paths
        if model_checkpoint_path is None:
            model_checkpoint_path = os.path.join(
                EYE_TRACKING_PATH,
                "inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
            )
        
        if model_config_path is None:
            model_config_path = os.path.join(
                EYE_TRACKING_PATH,
                "inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
            )
        
        self.model_checkpoint_path = model_checkpoint_path
        self.model_config_path = model_config_path
        self.device = device
        
        # Load config
        with open(self.model_config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        # Build and load model
        self.model = self._build_model()
        print(f"✅ Eye tracking model loaded on {self.device}")
    
    def _build_model(self):
        """Build the SocialEye model."""
        from inference.model.model_archs import MODEL_ARCH
        
        # Build backbone
        stage_info = MODEL_ARCH[self.config["MODEL"]["arch"]]
        kwargs = {k: v for k, v in self.config["MODEL"]["BACKBONE"].items() if k != "type"}
        kwargs["in_channels"] = 1
        
        from inference.model.backbone import SocialEye
        model_backbone = SocialEye(stage_info=stage_info, **kwargs)
        
        # Build head
        head_in_channel = model_backbone.out_channels
        head_out_channel = 2
        final_height_width = tuple(self.config["MODEL"]["HEAD"]["final_height_width"])
        
        head = SocialEyePredictionBoundHead(
            head_in_channel, head_out_channel, final_height_width
        )
        
        # Combine into model
        model = SocialEyeModel(backbone=model_backbone, head=head)
        
        # Load weights
        load_checkpoint(model, self.model_checkpoint_path)
        
        # Move to device and set to eval mode
        if torch.cuda.is_available() and self.device != "cpu":
            model = model.cuda(self.device)
            torch.backends.cudnn.benchmark = True
        
        model.eval()
        return model
    
    def preprocess_batch(self, eye_images):
        """
        Preprocess a batch of eye images.
        
        Args:
            eye_images: numpy array of shape (bsz, 240, 640, 3) or (bsz, 240, 640)
                       RGB/BGR or grayscale images with left and right eyes side-by-side
        
        Returns:
            torch.Tensor of shape (bsz, 2, 240, 320) ready for model input
        """
        bsz = eye_images.shape[0]
        h, w = 240, 640
        
        # Convert to grayscale if needed
        if eye_images.ndim == 4 and eye_images.shape[3] == 3:
            # RGB/BGR to grayscale: simple average
            eye_images_gray = eye_images.mean(axis=3)
        else:
            eye_images_gray = eye_images
        
        # Ensure shape is (bsz, 240, 640)
        assert eye_images_gray.shape == (bsz, h, w), \
            f"Expected shape ({bsz}, {h}, {w}), got {eye_images_gray.shape}"
        
        # Convert to torch tensor
        eye_images_tensor = torch.from_numpy(eye_images_gray).float().to(self.device)
        
        # Process each image in batch
        processed_batch = torch.zeros((bsz, 2, 240, 320), device=self.device)
        
        for i in range(bsz):
            img = eye_images_tensor[i]  # (240, 640)
            
            # Split into left and right eyes
            left_eye = img[:, :320]   # Left half
            right_eye = img[:, 320:]  # Right half
            
            # Resize and normalize left eye
            left_processed = self._resize_and_normalize(left_eye, should_flip=False)
            
            # Resize and normalize right eye (with flip)
            right_processed = self._resize_and_normalize(right_eye, should_flip=True)
            
            processed_batch[i, 0, :, :] = left_processed
            processed_batch[i, 1, :, :] = right_processed
        
        return processed_batch
    
    def _resize_and_normalize(self, image, should_flip=False):
        """
        Resize and normalize a single eye image.
        
        Args:
            image: torch.Tensor of shape (H, W)
            should_flip: Whether to flip horizontally
        
        Returns:
            torch.Tensor of shape (240, 320)
        """
        # Normalize to [0, 1] then shift to [-0.5, 0.5]
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8) - 0.5
        
        # Flip if needed (for right eye)
        if should_flip:
            image_norm = torch.fliplr(image_norm)
        
        # Resize to (240, 320)
        from torchvision import transforms
        resize_transform = transforms.Resize((240, 320))
        
        # Add batch and channel dims for resize
        image_resized = resize_transform(image_norm.unsqueeze(0).unsqueeze(0))
        
        return image_resized.squeeze(0).squeeze(0)
    
    @torch.no_grad()
    def predict(self, eye_images):
        """
        Run inference on a batch of eye images.
        
        Args:
            eye_images: numpy array of shape (bsz, 240, 640, 3) or (bsz, 240, 640)
        
        Returns:
            dict with keys:
                'yaw': (bsz,) array of yaw angles in radians
                'pitch': (bsz,) array of pitch angles in radians
                'yaw_lower': (bsz,) array of yaw lower bounds
                'pitch_lower': (bsz,) array of pitch lower bounds
                'yaw_upper': (bsz,) array of yaw upper bounds
                'pitch_upper': (bsz,) array of pitch upper bounds
        """
        # Preprocess
        processed_batch = self.preprocess_batch(eye_images)
        
        # Forward pass
        outputs = self.model.forward(processed_batch)
        
        # Post-process (denormalize if needed)
        stats = self.config["STATS"]
        mean = torch.tensor(stats["mA"], device=self.device)
        std = torch.tensor(stats["sA"], device=self.device)
        
        preds_main = outputs["main"] * std + mean
        preds_lower = outputs["lower"]
        preds_upper = outputs["upper"]
        
        # Convert to numpy
        preds_main = preds_main.detach().cpu().numpy()  # (bsz, 2)
        preds_lower = preds_lower.detach().cpu().numpy()
        preds_upper = preds_upper.detach().cpu().numpy()
        
        return {
            'yaw': preds_main[:, 0],
            'pitch': preds_main[:, 1],
            'yaw_lower': preds_lower[:, 0],
            'pitch_lower': preds_lower[:, 1],
            'yaw_upper': preds_upper[:, 0],
            'pitch_upper': preds_upper[:, 1],
        }
    
    def gaze_to_2d_coordinates(
        self, 
        yaw, 
        pitch, 
        image_width=1408, 
        image_height=1408,
        fov_horizontal=150.0,
        fov_vertical=150.0,
        depth_m=1.0
    ):
        """
        Convert yaw/pitch gaze angles to approximate 2D pixel coordinates.
        
        This is a simplified projection without full camera calibration.
        For accurate results, use camera intrinsics and extrinsics from VRS files.
        
        Args:
            yaw: (bsz,) array of yaw angles in radians
            pitch: (bsz,) array of pitch angles in radians
            image_width: Width of the target RGB image
            image_height: Height of the target RGB image
            fov_horizontal: Horizontal field of view in degrees (Aria RGB ~150°)
            fov_vertical: Vertical field of view in degrees
            depth_m: Assumed depth in meters (default 1.0m)
        
        Returns:
            numpy array of shape (bsz, 2) with [x, y] pixel coordinates
        """
        bsz = len(yaw)
        
        # Convert FOV to radians
        fov_h_rad = np.deg2rad(fov_horizontal)
        fov_v_rad = np.deg2rad(fov_vertical)
        
        # Simple pinhole camera projection
        # Assume camera center is at image center
        cx = image_width / 2.0
        cy = image_height / 2.0
        
        # Focal lengths (approximation from FOV)
        fx = image_width / (2.0 * np.tan(fov_h_rad / 2.0))
        fy = image_height / (2.0 * np.tan(fov_v_rad / 2.0))
        
        # Convert yaw/pitch to 3D direction vector
        # In CPF (Central Pupil Frame): X=down, Y=left, Z=forward
        # Yaw: rotation around vertical axis (left/right)
        # Pitch: rotation around horizontal axis (up/down)
        
        # Direction vector in camera frame
        x_cam = depth_m * np.tan(yaw)      # Horizontal displacement
        y_cam = depth_m * np.tan(pitch)    # Vertical displacement
        z_cam = depth_m                     # Depth
        
        # Project to image coordinates
        u = fx * (x_cam / z_cam) + cx
        v = fy * (y_cam / z_cam) + cy
        
        # Stack into (bsz, 2)
        coordinates_2d = np.stack([u, v], axis=1)
        
        # Clip to image bounds
        coordinates_2d[:, 0] = np.clip(coordinates_2d[:, 0], 0, image_width - 1)
        coordinates_2d[:, 1] = np.clip(coordinates_2d[:, 1], 0, image_height - 1)
        
        return coordinates_2d
    


class AriaGazeProjector:
    """
    Project gaze (yaw, pitch) to 2D image coordinates using device calibration.
    """
    
    def __init__(self, device_calibration=None, vrs_path= None):
        """
        Initialize the projector.
        
        Args:
            device_calibration: ProjectAria DeviceCalibration object
            vrs_path: Path to VRS file to load calibration from
        """
        self.device_calibration = device_calibration
        
        # If VRS path provided, load calibration
        if vrs_path is not None and device_calibration is None:
            self.load_calibration_from_vrs(vrs_path)
        
        # Check if calibration is available
        if self.device_calibration is None:
            print("[WARNING] No device calibration available. Will use simple projection.")
    
    def load_calibration_from_vrs(self, vrs_path: str):
        """Load device calibration from VRS file."""
        try:
            from projectaria_tools.core import data_provider
            
            print(f"[INFO] Loading calibration from VRS: {vrs_path}")
            provider = data_provider.create_vrs_data_provider(vrs_path)
            self.device_calibration = provider.get_device_calibration()
            print("[INFO] ✓ Device calibration loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load calibration from VRS: {e}")
            self.device_calibration = None
    
    def load_calibration_from_json(self, json_path: str):
        """Load device calibration from JSON file."""
        try:
            from projectaria_tools.core.calibration import device_calibration_from_json_string
            
            print(f"[INFO] Loading calibration from JSON: {json_path}")
            with open(json_path, 'r') as f:
                calib_json = f.read()
            
            self.device_calibration = device_calibration_from_json_string(calib_json)
            print("[INFO] ✓ Device calibration loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load calibration from JSON: {e}")
            self.device_calibration = None
    
    def project_gaze_accurate(
        self,
        yaw: float,
        pitch: float,
        depth_m: float = 1.0,
    ):
        """
        Project gaze to 2D image coordinates using accurate projection.
        
        Args:
            yaw: Gaze yaw angle in radians
            pitch: Gaze pitch angle in radians
            depth_m: Assumed depth in meters (default: 1.0)
        
        Returns:
            Tuple of (x, y) pixel coordinates, or None if projection fails
        """
        if self.device_calibration is None:
            print("[ERROR] Device calibration not available")
            return None
        
        try:
            from projectaria_tools.core.mps import EyeGaze
            from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
            
            # Create EyeGaze object
            eye_gaze = EyeGaze
            eye_gaze.yaw = yaw
            eye_gaze.pitch = pitch
            
            # Get RGB camera calibration
            rgb_stream_label = "camera-rgb"
            rgb_camera_calib = self.device_calibration.get_camera_calib(rgb_stream_label)
            
            # Project gaze to image coordinates
            gaze_projection = get_gaze_vector_reprojection(
                eye_gaze,
                rgb_stream_label,
                self.device_calibration,
                rgb_camera_calib,
                depth_m=depth_m,
            )
            
            if gaze_projection is not None:
                return (float(gaze_projection[0]), float(gaze_projection[1]))
            else:
                return None
                
        except Exception as e:
            print(f"[ERROR] Accurate projection failed: {e}")
            return None
    
    def project_gaze_batch(
        self,
        yaw_array: np.ndarray,
        pitch_array: np.ndarray,
        depth_m: float = 1.0,
    ) -> np.ndarray:
        """
        Project batch of gaze data to 2D coordinates.
        
        Args:
            yaw_array: Array of yaw angles (N,)
            pitch_array: Array of pitch angles (N,)
            depth_m: Assumed depth in meters
        
        Returns:
            Array of (x, y) coordinates, shape (N, 2)
        """
        n_samples = len(yaw_array)
        coordinates = np.zeros((n_samples, 2), dtype=np.float32)
        
        print(f"[INFO] Projecting {n_samples} gaze points...")
        
        for i in range(n_samples):
            result = self.project_gaze_accurate(
                yaw=yaw_array[i],
                pitch=pitch_array[i],
                depth_m=depth_m
            )
            
            if result is not None:
                coordinates[i, 0] = result[0]
                coordinates[i, 1] = result[1]
            else:
                # Fallback: use simple projection
                coordinates[i, 0] = 704  # Center
                coordinates[i, 1] = 704
                print(f"[WARNING] Frame {i}: projection failed, using center")
        
        print(f"[INFO] ✓ Projection complete")
        return coordinates
    
    def project_with_simple_fallback(
        self,
        yaw_array: np.ndarray,
        pitch_array: np.ndarray,
        image_width: int = 1408,
        image_height: int = 1408,
        depth_m: float = 1.0,
    ):
        """
        Project gaze with simple fallback if calibration is not available.
        
        Returns:
            Tuple of (coordinates_2d, is_accurate_mask) where:
                - coordinates_2d: (N, 2) array of pixel coordinates
                - is_accurate_mask: (N,) boolean array indicating accurate projection
        """
        n_samples = len(yaw_array)
        coordinates = np.zeros((n_samples, 2), dtype=np.float32)
        is_accurate = np.zeros(n_samples, dtype=bool)
        
        if self.device_calibration is not None:
            # Use accurate projection
            print("[INFO] Using accurate projection with device calibration")
            for i in range(n_samples):
                result = self.project_gaze_accurate(yaw_array[i], pitch_array[i], depth_m)
                if result is not None:
                    coordinates[i] = result
                    is_accurate[i] = True
                else:
                    # Simple fallback for this frame
                    coordinates[i] = self._simple_projection(
                        yaw_array[i], pitch_array[i], image_width, image_height
                    )
        else:
            # Use simple projection for all
            print("[INFO] Using simple projection (no calibration)")
            for i in range(n_samples):
                coordinates[i] = self._simple_projection(
                    yaw_array[i], pitch_array[i], image_width, image_height
                )
        
        return coordinates, is_accurate
    
    def _simple_projection(
        self,
        yaw: float,
        pitch: float,
        img_width: int,
        img_height: int,
        scale_factor: float = 800.0
    ) -> np.ndarray:
        """Simple linear projection (fallback)."""
        center_x = img_width / 2.0
        center_y = img_height / 2.0
        x = center_x + yaw * scale_factor
        y = center_y - pitch * scale_factor
        
        # Clamp to image bounds
        x = np.clip(x, 0, img_width - 1)
        y = np.clip(y, 0, img_height - 1)
        
        return np.array([x, y], dtype=np.float32)
    


def extract_gaze_point(eye_images, rgb_images):
    """
    eye_images: numpy array of shape (bsz, 240, 640, 3)
    """
    print("\n2. Loading eye tracking model...")
    model = EyeGazeInferenceBatch(device="cuda" if torch.cuda.is_available() else "cpu")

    print("\n3. Running inference...")
    results = model.predict(eye_images)

    print("\n4. Converting to 2D pixel coordinates...")

    # 🔑 IMPORTANT: Provide path to the VRS file that was used to collect this data
    vrs_path = "/home/alin/projects/piper_project/vrs_dongkyu.vrs"  # ← Replace with actual VRS file path

    # Initialize projector with calibration
    projector = AriaGazeProjector(vrs_path=vrs_path)

    # Project with calibration (with fallback to simple projection if VRS not found)
    coordinates_2d, is_accurate = projector.project_with_simple_fallback(
        yaw_array=results['yaw'],
        pitch_array=results['pitch'],
        image_width=rgb_images.shape[2],   # 1408
        image_height=rgb_images.shape[1],  # 1408
        depth_m=1.0  # Assume objects at 1 meter distance
    )

    return coordinates_2d



def overlay_gaze_on_image(rgb_image, x_coord, y_coord, copy_image=True, 
                          crosshair_size=20, crosshair_thickness=2, 
                          circle_radius=10, circle_thickness=2,
                          color=(0, 255, 0)):
    """
    Draw a gaze marker (crosshair + circle) on the image.
    
    Args:
        rgb_image: numpy array of shape (H, W, 3) - RGB or BGR format
        x_coord: x coordinate of the gaze point (can be float)
        y_coord: y coordinate of the gaze point (can be float)
        copy_image: If True, create a copy of the image (default: True)
        crosshair_size: Size of the crosshair arms (default: 20)
        crosshair_thickness: Thickness of crosshair lines (default: 2)
        circle_radius: Radius of the center circle (default: 10)
        circle_thickness: Thickness of the circle (default: 2)
        color: Color in RGB format (default: (0, 255, 0) = green)
               Note: OpenCV uses BGR, so this will be converted
    
    Returns:
        Image with gaze marker drawn (numpy array)
    """
    # Create copy if requested
    if copy_image:
        image = rgb_image.copy()
    else:
        image = rgb_image
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Convert coordinates to integers and clamp to image bounds
    x = int(np.clip(x_coord, 0, w - 1))
    y = int(np.clip(y_coord, 0, h - 1))
    
    # Convert RGB to BGR if using OpenCV (OpenCV uses BGR)
    # Assuming input is RGB, convert to BGR
    bgr_color = (color[2], color[1], color[0])
    
    # Calculate crosshair endpoints with boundary checks
    x_left = max(0, x - crosshair_size)
    x_right = min(w - 1, x + crosshair_size)
    y_top = max(0, y - crosshair_size)
    y_bottom = min(h - 1, y + crosshair_size)
    
    # Draw horizontal line
    cv2.line(image, (x_left, y), (x_right, y), bgr_color, crosshair_thickness)
    
    # Draw vertical line
    cv2.line(image, (x, y_top), (x, y_bottom), bgr_color, crosshair_thickness)
    
    # Draw center circle
    cv2.circle(image, (x, y), circle_radius, bgr_color, circle_thickness)
    
    # Optionally draw a small filled circle at the exact center
    cv2.circle(image, (x, y), 2, bgr_color, -1)
    
    return image