from builder import *
    
def main():
    print("===== BẮT ĐẦU QUÁ TRÌNH CHUẨN BỊ DỮ LIỆU =====")
    
    # build SQL database
    build_metadata_database()
    build_keyframes_database()
    build_objects_database()
    
    # build vector database
    build_clip_vector_store()
    build_keyword_vector_store()
    
    print("\n===== HOÀN TẤT QUÁ TRÌNH CHUẨN BỊ DỮ LIỆU =====")

if __name__ == "__main__":
    main()