import os
import glob
import json
import numpy as np
from qdrant_client import QdrantClient, models
from config.settings import settings
from tqdm import tqdm

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

def build_clip_vector_store():
    print("Bắt đầu xây dựng CLIP vector store với Qdrant...")
    vector_size = 512
    # create or recreate collection
    client.recreate_collection(
        collection_name=settings.QDRANT_VIDEO_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    print(f"-> Collection '{settings.QDRANT_VIDEO_COLLECTION_NAME}' đã được tạo/tái tạo.")
    
    # take all file .npy
    clip_feature_files = glob.glob(os.path.join(settings.RAW_CLIPFEATURE_DIR, '*.npy'))
    if not clip_feature_files:
        print(f"LỖI: Không tìm thấy file .npy nào trong thư mục: {settings.RAW_CLIPFEATURE_DIR}")
        return
    
    # use counter for each point id
    point_id_counter = 0
    batch_size = 500
    
    for file_path in tqdm(clip_feature_files):
        try:
            video_id = os.path.splitext(os.path.basename(file_path))[0]
            
            vectors = np.load(file_path)
            vectors = vectors.astype('float32')
            
            num_keyframes = vectors.shape[0]
            points_to_upload = []
            
            for i in range(num_keyframes):
                keyframe_id = f"{i:03d}"
                vector = vectors[i]
                
                point = models.PointStruct(
                    id=point_id_counter,
                    vector=vector.tolist(),
                    payload={
                        "video_id": video_id,
                        "keyframe_id": keyframe_id
                    }
                )
                points_to_upload.append(point)
                point_id_counter += 1
                
                # upsert when equal batch_size
                if len(points_to_upload) >= batch_size:
                    client.upsert(
                        collection_name=settings.QDRANT_VIDEO_COLLECTION_NAME,
                        points=points_to_upload,
                        wait=True
                    )
                    # print(f"-> Đã upload {len(points_to_upload)} điểm (batch).")
                    points_to_upload = []
                    
            # upload the rest
            if points_to_upload:
                client.upsert(
                    collection_name=settings.QDRANT_VIDEO_COLLECTION_NAME,
                    points=points_to_upload,
                    wait=True
                )
                # print(f"-> Đã upload {len(points_to_upload)} điểm (batch cuối).")
                
        except Exception as e:
            print(f"CẢNH BÁO: Bỏ qua file {file_path} do lỗi: {e}")
            continue
    
    print("Hoàn tất upload tất cả feature ✅")

from sentence_transformers import SentenceTransformer

def build_keyword_vector_store():
    print("Bắt đầu xây dựng keyword vector store với Qdrant...")

    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vector_size = embedding_model.get_sentence_embedding_dimension()
    client.recreate_collection(
        collection_name=settings.QDRANT_KEYWORD_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    print(f"-> Collection '{settings.QDRANT_KEYWORD_COLLECTION_NAME}' đã được tạo/tái tạo.")

    metadata_files = glob.glob(os.path.join(settings.RAW_METADATA_DIR, "*.json"))
    if not metadata_files:
        print(f"LỖI: Không tìm thấy file .json nào trong thư mục: {settings.RAW_METADATA_DIR}")
        return

    point_id_counter = 0
    batch_size = 500
    points_to_upload = set()

    for file_path in tqdm(metadata_files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            keywords = data.get("keywords", [])
            for keyword in keywords:
                embedding_vector = embedding_model.encode(keyword)

                point = models.PointStruct(
                    id=point_id_counter,
                    vector=embedding_vector,
                    payload={"keyword": keyword, "source_file": os.path.basename(file_path)}
                )
                points_to_upload.update(point)
                point_id_counter += 1

                if len(points_to_upload) >= batch_size:
                    client.upsert(
                        collection_name=settings.QDRANT_KEYWORD_COLLECTION_NAME,
                        points=list(points_to_upload),
                        wait=True
                    )
                    # print(f"-> Đã upload {len(points_to_upload)} điểm (batch).")
                    points_to_upload = []

        except Exception as e:
            print(f"CẢNH BÁO: Bỏ qua file {file_path} do lỗi: {e}")
            continue

    if points_to_upload:
        client.upsert(
            collection_name=settings.QDRANT_KEYWORD_COLLECTION_NAME,
            points=points_to_upload,
            wait=True
        )
        # print(f"-> Đã upload {len(points_to_upload)} điểm (batch cuối).")

    print("Hoàn tất upload tất cả keywords ✅")