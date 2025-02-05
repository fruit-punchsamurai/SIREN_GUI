import torch
from torch import nn
import numpy as np
from IPython.display import Audio
import os
import matplotlib.pyplot as plt
import cv2
import noisereduce as nr
import soundfile as sf
import subprocess
import shutil
import json
import argparse

parser = argparse.ArgumentParser(description="Run SIREN model inference.")
parser.add_argument('--model_path', type=str, required=True, help="Path to the .pth model file.")
parser.add_argument('--json_file', type=str, required=True, help="Path to the JSON file containing video parameters.")
parser.add_argument('--output_file', type=str, required=True, help="Path for the final output video.")

args = parser.parse_args()

def get_mgrid(sidelen, dim=2, super_resolution_factor=1):
    """Generates a flattened grid of (x, y, ...) coordinates in a range of -1 to 1, with super resolution."""
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    # If super resolution is greater than 1, generate finer grid
    if super_resolution_factor > 1:
        # Generate finer grid by increasing number of points
        sidelen = [dim_size * super_resolution_factor for dim_size in sidelen]

    if dim == 1:
        pixel_coords = np.linspace(0, 1, sidelen[0]).astype(np.float32)
        pixel_coords = pixel_coords[:, None]  # Reshape to (N, 1)
    elif dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError(f"Not implemented for dim={dim}")

    pixel_coords -= 0.5
    pixel_coords *= 2.0
    return torch.Tensor(pixel_coords).view(-1, dim)


# SIREN-based model
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, is_first_initialization=1):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_first_initialization = is_first_initialization
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_( -self.is_first_initialization/ self.in_features, self.is_first_initialization / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class SharedSiren(nn.Module):
    def __init__(self, 
                 audio_initial_features, audio_initial_layers, 
                 video_initial_features, video_initial_layers, 
                 shared_hidden_features, shared_hidden_layers, 
                 video_hidden_features, video_hidden_layers, 
                 audio_hidden_features, audio_hidden_layers, hidden_omega_0=30.0,
                 siamese_audio=False, audio_hidden_features_siam=0, audio_hidden_layers_siam=0):
        super().__init__()

        self.siamese_audio = siamese_audio

        # Audio initial branch
        self.audio_initial_branch = []
        self.audio_initial_branch.append(SineLayer(1, audio_initial_features, is_first=True, omega_0=100, is_first_initialization=25))  
        for _ in range(audio_initial_layers - 1):
            self.audio_initial_branch.append(SineLayer(audio_initial_features, audio_initial_features, is_first=False, omega_0=hidden_omega_0))
        self.audio_initial_branch = nn.Sequential(*self.audio_initial_branch)

        # Video initial branch
        self.video_initial_branch = []
        self.video_initial_branch.append(SineLayer(3, video_initial_features, is_first=True, omega_0=100, is_first_initialization=2))  
        for _ in range(video_initial_layers - 1):
            self.video_initial_branch.append(SineLayer(video_initial_features, video_initial_features, is_first=False, omega_0=hidden_omega_0))
        self.video_initial_branch = nn.Sequential(*self.video_initial_branch)

        # Shared layers
        shared_input_features = audio_initial_features + video_initial_features
        self.shared_layers = []
        self.shared_layers.append(SineLayer(shared_input_features, shared_hidden_features, is_first=False, omega_0=hidden_omega_0))
        for _ in range(shared_hidden_layers - 1):
            self.shared_layers.append(SineLayer(shared_hidden_features, shared_hidden_features, is_first=False, omega_0=hidden_omega_0))
        self.shared_layers = nn.Sequential(*self.shared_layers)

        # Video branch
        self.video_branch = []
        self.video_branch.append(SineLayer(shared_hidden_features, video_hidden_features, is_first=False, omega_0=hidden_omega_0))
        for _ in range(video_hidden_layers - 1):
            self.video_branch.append(SineLayer(video_hidden_features, video_hidden_features, is_first=False, omega_0=hidden_omega_0))
        self.video_branch.append(nn.Linear(video_hidden_features, 3))  # Output RGB
        self.video_branch = nn.Sequential(*self.video_branch)

        # Audio branch
        self.audio_branch = []
        self.audio_branch.append(SineLayer(shared_hidden_features, audio_hidden_features, is_first=False, omega_0=hidden_omega_0))
        for _ in range(audio_hidden_layers - 1):
            self.audio_branch.append(SineLayer(audio_hidden_features, audio_hidden_features, is_first=False, omega_0=hidden_omega_0))
        self.audio_branch.append(nn.Linear(audio_hidden_features, 1))  # Output amplitude
        self.audio_branch = nn.Sequential(*self.audio_branch)

        if self.siamese_audio:
            self.audio_branch_siam = []
            self.audio_branch_siam.append(SineLayer(shared_hidden_features, audio_hidden_features_siam, 
                                                    is_first=False, omega_0=hidden_omega_0))
            for _ in range(audio_hidden_layers_siam - 1):
                self.audio_branch_siam.append(SineLayer(audio_hidden_features_siam, audio_hidden_features_siam, 
                                                        is_first=False, omega_0=hidden_omega_0))
            self.audio_branch_siam.append(nn.Linear(audio_hidden_features_siam, 1))  # Output amplitude for audio
            self.audio_branch_siam = nn.Sequential(*self.audio_branch_siam)

    def forward(self, coords):
    # Remove batch dimension if present
        if coords.dim() == 3:  # Shape [1, N, 4]
            coords = coords.squeeze(0)  # Shape [N, 4]

        # Split coordinates into video and audio parts
        video_coords = coords[:, :3]  # First 3 dimensions for video (x, y, frameindex)
        audio_coords = coords[:, 3:]  # Last dimension for audio (t)


        # Pass inputs through initial branches
        video_initial_output = self.video_initial_branch(video_coords)  # Pass video coordinates to video branch
        audio_initial_output = self.audio_initial_branch(audio_coords)  # Pass audio coordinates to audio branch

        # Combine outputs from initial branches
        combined_output = torch.cat([video_initial_output, audio_initial_output], dim=-1)

        # Pass combined output through shared layers
        shared_output = self.shared_layers(combined_output)



        # Pass shared output through video and audio branches
        video_output = self.video_branch(shared_output)
        audio_output = self.audio_branch(shared_output)

        if self.siamese_audio:
            audio_output_siam = self.audio_branch_siam(shared_output)
            return video_output, audio_output, audio_output_siam

        return video_output, audio_output
    


def get_coords_from_params(frames, height, width, audio_samples, super_resolution_factor=1):
    """
    Generate synchronized video and audio coordinates with super resolution for fine-grained grid sampling.

    Args:
        frames (int): Number of video frames.
        height (int): Height of each frame.
        width (int): Width of each frame.
        audio_samples (int): Number of audio samples.
        super_resolution_factor (int): Factor to generate finer grid points between coordinates.

    Returns:
        torch.Tensor: Combined coordinates [N, 4], where N is the total number of synchronized samples.
        dict: Metadata containing the original sizes of video and audio grids.
    """
    # Video coordinates (x, y, frameindex) with super resolution
    mgrid_video = get_mgrid((height, width), dim=2, super_resolution_factor=super_resolution_factor)  # Fine grid for one frame
    frameindex = torch.linspace(0, 1, frames).view(-1, 1).repeat_interleave(mgrid_video.shape[0], dim=0)
    mgrid_video = mgrid_video.repeat(frames, 1)
    mgrid_video = torch.cat([mgrid_video, frameindex], dim=1)  # Shape [frames * height * width, 3]

    # Audio coordinates (t)
    mgrid_audio = torch.linspace(0, 1, audio_samples).view(-1, 1)  # Shape [audio_samples, 1]

    # Synchronize video and audio coordinates
    video_size = mgrid_video.shape[0]
    audio_size = mgrid_audio.shape[0]

    original_lengths = {
        "video_size": video_size,
        "audio_size": audio_size,
    }

    if video_size < audio_size:
        # Repeat video to match audio size
        repeat_factor = audio_size // video_size
        remainder = audio_size % video_size
        mgrid_video = mgrid_video.repeat(repeat_factor, 1)
        if remainder > 0:
            mgrid_video = torch.cat([mgrid_video, mgrid_video[:remainder]], dim=0)
    elif audio_size < video_size:
        # Repeat audio to match video size
        repeat_factor = video_size // audio_size
        remainder = video_size % audio_size
        mgrid_audio = mgrid_audio.repeat(repeat_factor, 1)
        if remainder > 0:
            mgrid_audio = torch.cat([mgrid_audio, mgrid_audio[:remainder]], dim=0)

    # Combine synchronized video and audio coordinates
    coords = torch.cat([mgrid_video, mgrid_audio], dim=1)  # Shape [max(video_size, audio_size), 4]

    # Store the synchronization details
    original_lengths["synced_size"] = coords.shape[0]
    return coords, original_lengths


model_path = args.model_path
json_file = args.json_file

# Load the video_parameters from the JSON file
with open(json_file, "r") as f:
    video_parameters = json.load(f)

# Access the video_parameters from the loaded dictionary
height = video_parameters["height"]
width = video_parameters["width"]
channels = video_parameters["channels"]
num_frames = video_parameters["num_frames"]
fps = video_parameters["fps"]
audio_rate = video_parameters["audio_rate"]
audio_duration = video_parameters["audio_duration"]
frame_size = height * width
audio_sample = audio_rate * audio_duration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SharedSiren(
    audio_initial_features=512,
    audio_initial_layers=1,
    video_initial_features=512,
    video_initial_layers=1,
    shared_hidden_features=512,
    shared_hidden_layers=5,
    video_hidden_features=512,
    video_hidden_layers=1,
    audio_hidden_features=512,
    audio_hidden_layers=1,
    hidden_omega_0=30,
    siamese_audio=True,
    audio_hidden_features_siam=512,
    audio_hidden_layers_siam=1
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))



# Generate coordinates and move to the correct device
coords, original_lengths = get_coords_from_params(num_frames, height, width, audio_sample)
coords = coords.to(device)


model_output_video = []
model_output_audio = []
model_output_siam = []


for i in range(num_frames):
    # Adjust to use the super-resolved frame size
    start_idx = i * height * width
    end_idx = start_idx + height * width
    frame_coords = coords[start_idx:end_idx, :].to(device)  # Ensure the coordinates are on the correct device

    # Pass the fine-grained frame through the model
    model_output_frame, model_audio_frame, model_output_siam_frame = model(frame_coords)

    # Convert to NumPy and append to the lists
    model_output_video.append(model_output_frame.detach().cpu().numpy())
    model_output_audio.append(model_audio_frame.detach().cpu().numpy())
    model_output_siam.append(model_output_siam_frame.detach().cpu().numpy())

# Stack the collected outputs into NumPy arrays
model_output_video = np.vstack(model_output_video)
model_output_audio = np.vstack(model_output_audio)
model_output_siam = np.vstack(model_output_siam)

# Trim the outputs to match original sizes
trimmed_video_output = model_output_video[:original_lengths['video_size'], :]
trimmed_audio_output = model_output_audio[:original_lengths['audio_size'], :]
trimmed_audio_siren = model_output_siam[:original_lengths['audio_size'], :]

# Denormalize the video output to [0, 1] range
output_video = (trimmed_video_output + 1) / 2  # Denormalize from [-1, 1] to [0, 1]

output_video = output_video.reshape(num_frames, height, width, channels)
noise=trimmed_audio_output-trimmed_audio_siren
audio_output = trimmed_audio_output
denoised_audio = nr.reduce_noise(y=audio_output.squeeze(), sr=audio_rate, y_noise=noise.squeeze(), stationary=True)

# Convert the denoised audio back to a tensor and move to the correct device
denoised_audio_tensor = torch.tensor(denoised_audio, device=device)

temp_frames_dir = "temp_frames"
temp_video_path = "temp_video.mp4"  # Temporary video file path
temp_audio_path = "temp_audio.wav"  # Temporary audio file path
final_output_path = args.output_file


# Create the directory for temp frames if it doesn't exist
os.makedirs(temp_frames_dir, exist_ok=True)

output_videoo = (output_video * 255).clip(0, 255).astype(np.uint8)
output_dir = os.path.dirname(final_output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
# Step 1: Save video frames as PNG (highest quality)
for i in range(num_frames):
    frame = output_videoo[i]  # Shape: [height, width, channels]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_filename = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")  # Save as PNG for highest quality
    cv2.imwrite(frame_filename, frame_rgb)

# Write audio to the temp file
sf.write(temp_audio_path, denoised_audio, audio_rate)

# Step 2: Combine video frames into a temporary video file
image_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])

# Check if there are any images in the directory
if not image_files:
    print("No images found in the directory.")
    exit()

# Get the frame size from the first image
first_image = cv2.imread(os.path.join(temp_frames_dir, image_files[0]))
height, width, _ = first_image.shape

# Use I420 or YUV420 for uncompressed video (you can also try MJPG for lossless but with some compression)
# fourcc = cv2.VideoWriter_fourcc(*'I420')  # Uncompressed video codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create VideoWriter object with the chosen codec
video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

# Iterate through all images and add them to the video
for image_file in image_files:
    image_path = os.path.join(temp_frames_dir, image_file)
    image = cv2.imread(image_path)
    
    if image is not None:
        video_writer.write(image)
    else:
        print(f"Skipping invalid image: {image_file}")

# Release the video writer and finalize the video
video_writer.release()


# Step 3: Combine video and audio into the final output
def combine_video_audio(video_file, audio_file, output_file):
    # FFmpeg command to combine video and audio
    command = [
        "ffmpeg", 
        "-y",
        "-i", video_file, 
        "-i", audio_file, 
        "-c:v", "copy", 
        "-c:a", "aac", 
        "-strict", "experimental", 
        output_file
    ]
    
    # Run the FFmpeg command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg process: {e}")

combine_video_audio(temp_video_path, temp_audio_path, final_output_path)

# Step 4: Cleanup - Delete temp files after the final video is created
def cleanup_temp_files():
    # Delete the frames directory and its contents
    shutil.rmtree(temp_frames_dir)
    
    # Delete temporary video and audio files
    os.remove(temp_video_path)
    
    os.remove(temp_audio_path)

# Cleanup the temporary files
cleanup_temp_files()

print(f"Final output video is saved at {final_output_path}")

