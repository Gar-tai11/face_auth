import os
import io
import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from supabase import create_client, ClientOptions
from PIL import Image

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

SUPABASE_URL              = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
API_KEY                   = os.getenv("API_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    options=ClientOptions(schema="attendance")
)

app = FastAPI()

# ============================================================
# 認証
# ============================================================
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ============================================================
# POST /register
# ============================================================
@app.post("/register")
async def register(
    user_id: str,
    image: UploadFile = File(...),
    _: None = Depends(verify_api_key)
):
    contents = await image.read()
    pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
    np_img   = np.array(pil_img)

    encodings = face_recognition.face_encodings(np_img)
    if not encodings:
        raise HTTPException(status_code=400, detail="顔が検出できませんでした")

    supabase.table("face_encodings").insert({
        "user_id":     user_id,
        "encoding":    encodings[0].tolist(),
        "is_adaptive": False
    }).execute()

    return JSONResponse({"status": "ok", "user_id": user_id})

# ============================================================
# GET /users
# ============================================================
@app.get("/users")
async def get_users(_: None = Depends(verify_api_key)):
    users_res = supabase.table("users").select(
        "supabase_auth_user_id, card_id"
    ).execute()

    enc_res = supabase.table("face_encodings").select("user_id").execute()

    count_map = {}
    for row in enc_res.data:
        uid = row["user_id"]
        count_map[uid] = count_map.get(uid, 0) + 1

    result = [
        {
            "user_id":        row["supabase_auth_user_id"],
            "card_id":        row["card_id"],
            "encoding_count": count_map.get(row["supabase_auth_user_id"], 0)
        }
        for row in users_res.data
    ]
    return JSONResponse(result)

# ============================================================
# DELETE /users/{user_id}/encodings
# ============================================================
@app.delete("/users/{user_id}/encodings")
async def delete_encodings(user_id: str, _: None = Depends(verify_api_key)):
    supabase.table("face_encodings").delete().eq("user_id", user_id).execute()
    return JSONResponse({"status": "ok", "user_id": user_id})
