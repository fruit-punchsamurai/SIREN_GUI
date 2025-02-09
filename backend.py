from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict
import animal_avatar
import random
import os
import subprocess
from pathlib import Path
import uuid

app = FastAPI()


# Create an empty avatars folder
if os.path.exists("avatars"):
    os.system("rm -rf avatars")
os.makedirs("avatars", exist_ok=True)

# Base directory for storing uploads and inference outputs
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


# Adjectives and Nouns for generating random usernames
adjectives = [
    "Brave", "Clever", "Mysterious", "Gentle", "Wicked", "Swift", "Ancient", "Lucky",
    "Crimson", "Silver", "Golden", "Fierce", "Quiet", "Daring", "Wistful", "Shadowy"
    ]
names = [
    "Fox", "Raven", "Storm", "Wolf", "Willow", "Ember", "Moon", "Falcon",
    "Thorn", "Viper", "Echo", "Hawk", "Lark", "Orchid", "Drake", "Zephyr"
    ]

# All mounts for serving static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/avatars", StaticFiles(directory="avatars"), name="avatars")

# Mount the uploads directory so that files can be served statically
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


# Store connected clients
clients: Dict[str, WebSocket] = {}
uniqueNames: Dict[str, str] = {}

# WebRTC configuration (using Google's public STUN server)
WEBRTC_CONFIG = {
    "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get():
    return HTMLResponse(content=open("pages/index.html").read())

@app.get("/infer", response_class=HTMLResponse)
async def read_infer():
    return HTMLResponse(content=open("pages/infer.html").read())


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()

    # Generate a unique name and avatar for the client
    unique_name = random.choice(adjectives) + random.choice(names)
    avatarSvg = animal_avatar.Avatar(unique_name, size=200).create_avatar()
    os.makedirs("avatars", exist_ok=True)
    with open(f"avatars/{client_id}.svg", "w") as f:
        f.write(avatarSvg)


    clients[client_id] = websocket
    uniqueNames[client_id] = unique_name
    print(f"Client {client_id} connected")

    try:
        # Assign identity to the client
        await websocket.send_json({
            "event": "assignIdentity",
            "clientId": client_id,
            "clientName": unique_name,
            "avatarSvg": avatarSvg
        })

        # Notify all clients about updated list
        await broadcast_clients_list()

        while True:
            data = await websocket.receive_json()
            event_type = data.get("event")

            if event_type == "offer":
                # Forward offer to target client
                target_client = clients.get(data["targetId"])
                if target_client:
                    await target_client.send_json({
                        "event": "offer",
                        "senderId": client_id,
                        "offer": data["offer"]
                    })

            elif event_type == "answer":
                # Forward answer to initiating client
                initiating_client = clients.get(data["targetId"])
                if initiating_client:
                    await initiating_client.send_json({
                        "event": "answer",
                        "senderId": client_id,
                        "answer": data["answer"]
                    })

            elif event_type == "ice-candidate":
                # Forward ICE candidate to target client
                target_client = clients.get(data["targetId"])
                if target_client:
                    await target_client.send_json({
                        "event": "ice-candidate",
                        "senderId": client_id,
                        "candidate": data["candidate"]
                    })

            elif event_type == "transferRequest":
                # Forward transfer request to target client
                target_client = clients.get(data["targetId"])
                if target_client:
                    await target_client.send_json({
                        "event": "transferRequest",
                        "senderId": client_id,
                        "fileName": data["fileName"],
                        "fileSize": data["fileSize"]
                    })

            elif event_type == "transferResponse":
                # Forward transfer response to initiating client
                initiating_client = clients.get(data["targetId"])
                if initiating_client:
                    await initiating_client.send_json({
                        "event": "transferResponse",
                        "senderId": client_id,
                        "accepted": data["accepted"]
                    })

    except WebSocketDisconnect:
        del clients[client_id]
        del uniqueNames[client_id]
        os.remove(f"avatars/{client_id}.svg")
        await broadcast_clients_list()
        print(f"Client {client_id} disconnected")

async def broadcast_clients_list():
    for client_id, websocket in clients.items():
        try:
            await websocket.send_json({
                "event": "updateClients",
                "uniqueNames": uniqueNames,
            })
        except:
            pass


@app.post("/upload_and_infer")
async def upload_and_infer(
    folder: str = Form(...),
    model_file: UploadFile = File(...)
):
    """
    Uploads a model file into a specified folder,
    then runs inference using the file and saves the output video
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
    
    # Generate a unique name for the output video
    output_video_name = f"output_{Path(model_file.filename).stem}_{uuid.uuid4().hex}.webm"
    output_video_path = os.path.join(folder_path, output_video_name)
    
    # Build the command to run the inference script
    model_file_path = os.path.abspath(model_file_path)
    output_video_path = os.path.abspath(output_video_path)
    command = [
        "python", "infer.py",
        "--model_path", model_file_path,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
