from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import subprocess
from pathlib import Path
import uuid

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Base directory for storing uploads and inference outputs
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


@app.post("/upload_and_infer")
async def upload_and_infer(
    folder: str = Form(...),
    model_file: UploadFile = File(...),
    json_file: UploadFile = File(...)
):
    """
    Uploads a model file and a JSON file into a specified folder,
    then runs inference using these files and saves the output video
    in the same folder.
    """
    folder = folder.strip()
    # Create the folder for this upload if it doesn't exist
    folder_path = os.path.join(UPLOADS_DIR, folder)
    os.makedirs(folder_path, exist_ok=True)
    
    # Save the model file
    model_file_path = os.path.join(folder_path, model_file.filename)
    try:
        with open(model_file_path, "wb") as f:
            f.write(await model_file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving model file: {e}")
    
    # Save the JSON file
    json_file_path = os.path.join(folder_path, json_file.filename)
    try:
        with open(json_file_path, "wb") as f:
            f.write(await json_file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving JSON file: {e}")
    
    # Generate a unique name for the output video
    output_video_name = f"output_{Path(model_file.filename).stem}_{uuid.uuid4().hex}.webm"
    output_video_path = os.path.join(folder_path, output_video_name)
    
    # Build the command to run the inference script
    model_file_path = os.path.abspath(model_file_path)
    json_file_path = os.path.abspath(json_file_path)
    output_video_path = os.path.abspath(output_video_path)
    command = [
        "python", "infer.py",
        "--model_path", model_file_path,
        "--json_file", json_file_path,
        "--output_file", output_video_path  # Make sure infer.py accepts this argument.
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Inference stdout:", result.stdout)
        print("Inference stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Subprocess error output:", e.stderr)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e.stderr}")
    
    # Return the URL to access the generated video.
    # The video will be served from the /uploads endpoint.
    video_url = f"/uploads/{folder}/{output_video_name}"
    return {"video_url": video_url}

# @app.get("/list_all_videos")
# async def list_all_videos():
#     """
#     Lists all subfolders in the uploads directory and the MP4 videos within each folder.
#     """
#     uploads_listing = []
#     for folder in os.listdir(UPLOADS_DIR):
#         folder_path = os.path.join(UPLOADS_DIR, folder)
#         if os.path.isdir(folder_path):
#             videos = [f for f in os.listdir(folder_path) if f.endswith(".webm")]
#             uploads_listing.append({
#                 "folder": folder,
#                 "videos": videos
#             })
#     return {"uploads": uploads_listing}

@app.get("/list_all_videos")
async def list_all_videos():
    """
    Lists all subfolders in the uploads directory and the WebM videos within each folder.
    """
    uploads_listing = []
    for folder in os.listdir(UPLOADS_DIR):
        folder_path = os.path.join(UPLOADS_DIR, folder)
        if os.path.isdir(folder_path):
            videos = [f for f in os.listdir(folder_path) if f.endswith(".webm")]  # Ensure filtering for .webm
            uploads_listing.append({
                "folder": folder,
                "videos": videos
            })
    return {"uploads": uploads_listing}



# @app.get("/videos/{folder}")
# async def list_videos(folder: str):
#     # Log for debugging
#     folder = folder.strip()  # Optional: strip any accidental spaces
#     folder_path = os.path.join(UPLOADS_DIR, folder)
#     print("GET /videos/ endpoint: folder =", repr(folder), "->", os.path.abspath(folder_path))
#     if not os.path.exists(folder_path):
#         raise HTTPException(status_code=404, detail="Folder not found")
#     videos = [f for f in os.listdir(folder_path) if f.endswith(".webm")]
#     return {"videos": videos}

@app.get("/videos/{folder}")
async def list_videos(folder: str):
    folder = folder.strip()  # Remove any extra whitespace
    folder_path = os.path.join(UPLOADS_DIR, folder)
    
    print("GET /videos/ endpoint: folder =", repr(folder), "->", os.path.abspath(folder_path))
    
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    
    videos = [f for f in os.listdir(folder_path) if f.endswith(".webm")]  # Ensure filtering for .webm
    return {"videos": videos}



@app.get("/videos/{folder}")
async def list_videos(folder: str):
    folder = folder.strip()  # Remove any extra whitespace
    folder_path = os.path.join(UPLOADS_DIR, folder)
    print("GET /videos/ endpoint: folder =", repr(folder), "->", os.path.abspath(folder_path))
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    videos = [f for f in os.listdir(folder_path) if f.endswith(".webm")]
    return {"videos": videos}



# Mount the uploads directory so that files can be served statically
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
