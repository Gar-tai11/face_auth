import os
import time
import subprocess
import numpy as np
import cv2
import face_recognition
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY    = os.getenv("SUPABASE_ANON_KEY")
RECOGNITION_THRESHOLD = 0.45
COOLDOWN_SECONDS      = 5
SCALE_FACTOR          = 0.25
ALPHA                 = 0.05

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

user_data: dict[str, dict] = {}

def load_from_supabase():
    print("[起動] Supabaseからデータを読み込み中...")

    users_res = supabase.table("users").select("supabase_auth_user_id, card_id").execute()
    card_id_map = {
        row["supabase_auth_user_id"]: row["card_id"]
        for row in users_res.data
    }

    enc_res = supabase.table("face_encodings").select("user_id, id, encoding").execute()

    for row in enc_res.data:
        uid = row["user_id"]
        enc = np.array(row["encoding"], dtype=np.float64)

        if uid not in user_data:
            user_data[uid] = {
                "card_id":   card_id_map.get(uid, ""),
                "encodings": [],
                "enc_ids":   [] 
            }
        user_data[uid]["encodings"].append(enc)
        user_data[uid]["enc_ids"].append(row["id"])

    total = sum(len(v["encodings"]) for v in user_data.values())
    print(f"[起動] {len(user_data)}人 / {total}件のベクトルを読み込みました")

def update_encoding(user_id: str, enc_index: int, new_encoding: np.ndarray):
    stored  = user_data[user_id]["encodings"][enc_index]
    updated = (1 - ALPHA) * stored + ALPHA * new_encoding
    user_data[user_id]["encodings"][enc_index] = updated

    enc_id = user_data[user_id]["enc_ids"][enc_index]
    supabase.table("face_encodings").update({
        "encoding":    updated.tolist(),
        "is_adaptive": True
    }).eq("id", enc_id).execute()

def type_text(text: str):
    subprocess.run(["wtype", text])
    print(f"[入力] {text}")

def main():
    load_from_supabase()

    if not user_data:
        print("[エラー] 登録されたユーザーがいません。先に顔登録APIを呼び出してください。")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_input_time = {}

    print("カメラ映像を開始します。qキーで終了。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[エラー] カメラの読み取りに失敗しました")
            break

        small     = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encs      = face_recognition.face_encodings(rgb_small, face_locations)

        for face_enc, face_loc in zip(face_encs, face_locations):

            best_uid   = None
            best_idx   = None
            best_dist  = float("inf")

            for uid, data in user_data.items():
                if not data["encodings"]:
                    continue
                dists    = face_recognition.face_distance(data["encodings"], face_enc)
                min_idx  = int(np.argmin(dists))
                min_dist = dists[min_idx]
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_uid  = uid
                    best_idx  = min_idx

            scale = int(1 / SCALE_FACTOR)
            top, right, bottom, left = [v * scale for v in face_loc]

            if best_uid and best_dist < RECOGNITION_THRESHOLD:
                card_id = user_data[best_uid]["card_id"]
                now     = time.time()

                if now - last_input_time.get(best_uid, 0) > COOLDOWN_SECONDS:
                    print(f"[認証成功] user_id={best_uid} card_id={card_id} (距離: {best_dist:.3f})")
                    type_text(card_id)
                    last_input_time[best_uid] = now
                    update_encoding(best_uid, best_idx, face_enc)
                    label_color = (0, 255, 0)
                    label       = f"{card_id} OK"
                else:
                    label_color = (0, 165, 255)
                    label       = f"{card_id} 待機中"
            else:
                label_color = (0, 0, 255)
                label       = "Unknown"

            cv2.rectangle(frame, (left, top), (right, bottom), label_color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

        display = cv2.resize(frame, (960, 540))
        cv2.imshow("顔認証システム | qで終了", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
