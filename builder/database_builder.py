import sqlite3
import json
import glob
import os
from datetime import datetime
from config.settings import settings
from tqdm import tqdm
import pandas as pd

def build_metadata_database():
    # connect to database (create if not exists)
    print("Bắt đầu xây dựng metadata database...")
    conn = sqlite3.connect(settings.METADATA_KEYFRAME_OBJECT_DB_PATH)
    cursor = conn.cursor()
    
    # create table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        author TEXT,
        channel_id TEXT,
        channel_url TEXT,
        description TEXT,
        keywords TEXT,
        length INTEGER,
        publish_date TEXT,
        thumbnail_url TEXT,
        title TEXT,
        watch_url TEXT
    )
    ''')
    
    # take list of all metadata files
    metadata_files = glob.glob(os.path.join(settings.RAW_METADATA_DIR, '*.json'))
    if not metadata_files:
        print(f"LỖI: Không tìm thấy file metadata nào trong thư mục: {settings.RAW_METADATA_DIR}")
        return
    
    for file_path in tqdm(metadata_files):
        video_id = os.path.splitext(os.path.basename(file_path))[0] # taje video_id from filename
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # convert date
        try:
            # "01/08/2024" -> "2024-08-01"
            publish_date_obj = datetime.strptime(data.get("publish_date", ""), '%d/%m/%Y')
            formatted_date = publish_date_obj.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            formatted_date = None
            
        # combine all keywords into one string
        keywords_str = ", ".join([word.lower() for word in data.get("keywords", [])])

        cursor.execute(
            """
            INSERT OR REPLACE INTO videos (video_id, author, channel_id, channel_url, description, keywords, length, publish_date, thumbnail_url, title, watch_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                data.get("author").lower(),
                data.get("channel_id"),
                data.get("channel_url"),
                data.get("description").lower(),
                keywords_str,
                data.get("length"),
                formatted_date,
                data.get("thumbnail_url"),
                data.get("title").lower(),
                data.get("watch_url")
            )
        )
        
    conn.commit()
    conn.close()
    print(f"-> Đã xử lý {len(metadata_files)} file metadata. Xây dựng metadata database thành công!")
    
def build_keyframes_database():
    print("Bắt đầu xây dựng keyframe database...")
    conn = sqlite3.connect(settings.METADATA_KEYFRAME_OBJECT_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE keyframes (
        video_id TEXT,
        keyframe_id TEXT,
        pts_time FLOAT,
        frame_idx INTEGER,
        PRIMARY KEY (video_id, keyframe_id),
        FOREIGN KEY (video_id) REFERENCES videos(video_id)
    );         
    ''')
    
    map_keyframe_files = glob.glob(os.path.join(settings.RAW_MAP_KEYFRAME_DIR, '**', '*.csv'), recursive=True)
    if not map_keyframe_files:
        print(f"LỖI: Không tìm thấy file object nào trong thư mục: {settings.RAW_MAP_KEYFRAME_DIR}")
        return
    
    keyframes_to_insert = []
    for file_path in tqdm(map_keyframe_files):
        try:
            parts = file_path.replace('\\', '/').split('/')
            video_id = os.path.splitext(parts[-1])[0]
            
            df = pd.read_csv(file_path)
            pts_time = df['pts_time'].tolist()
            frame_idx = df['frame_idx'].tolist()
            
            for i in range(len(df)):
                keyframe_id = str(int(df['n'].iloc[i])).zfill(3)
                    
                keyframes_to_insert.append((
                    video_id,
                    keyframe_id,
                    pts_time[i],
                    frame_idx[i]
                ))
                
                
        except Exception as e:
            print(f"CẢNH BÁO: Bỏ qua file keyframe bị lỗi {file_path}. Lỗi: {e}")
            continue
    
    if keyframes_to_insert:
        cursor.executemany(
            """
            INSERT INTO keyframes (video_id, keyframe_id, pts_time, frame_idx)
            VALUES (?, ?, ?, ?)
            """,
            keyframes_to_insert
        )
    
    conn.commit()
    conn.close()
    print(f"-> Đã xử lý {len(map_keyframe_files)} file keyframe. Xây dựng keyframe database thành công!")
            
def build_objects_database():
    print("Bắt đầu xây dựng object database...")
    conn = sqlite3.connect(settings.METADATA_KEYFRAME_OBJECT_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS objects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT,
        keyframe_id TEXT,
        object_name TEXT,
        confidence REAL,
        ymin REAL,
        xmin REAL,
        ymax REAL,
        xmax REAL,
        FOREIGN KEY (video_id, keyframe_id) REFERENCES keyframes(video_id, keyframe_id)
    )
    ''')

    object_files = glob.glob(os.path.join(settings.RAW_OBJECT_DIR, '**', '*.json'), recursive=True)
    if not object_files:
        print(f"LỖI: Không tìm thấy file object nào trong thư mục: {settings.RAW_OBJECT_DIR}")
        return
        
    objects_to_insert = []
    for file_path in tqdm(object_files):
        try:
            parts = file_path.replace('\\', '/').split('/')
            video_id = parts[-2]
            keyframe_id = os.path.splitext(parts[-1])[0]

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # "Giải nén" cấu trúc dữ liệu
            scores = data.get("detection_scores", [])
            names = data.get("detection_class_entities", [])
            names = [n.lower() for n in names]
            boxes = data.get("detection_boxes", [])
            
            # Lặp qua từng đối tượng được phát hiện trong file
            for i in range(len(scores)):
                box = boxes[i]
                objects_to_insert.append((
                    video_id,
                    keyframe_id,
                    names[i],
                    float(scores[i]),
                    float(box[0]), # ymin
                    float(box[1]), # xmin
                    float(box[2]), # ymax
                    float(box[3])  # xmax
                ))
        except (json.JSONDecodeError, IndexError, KeyError, ValueError) as e:
            print(f"CẢNH BÁO: Bỏ qua file object bị lỗi {file_path}. Lỗi: {e}")
            continue

    if objects_to_insert:
        cursor.executemany(
            """
            INSERT INTO objects (video_id, keyframe_id, object_name, confidence, ymin, xmin, ymax, xmax)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            objects_to_insert
        )
    
    conn.commit()
    conn.close()
    print(f"-> Đã xử lý {len(object_files)} file object. Xây dựng object database thành công!")