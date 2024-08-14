import cv2
import mediapipe as mp
import cvzone

# Khởi tạo Face Mesh và Drawing utils của MediaPipe
mp_face_detection = mp.solutions.face_detection

# Sử dụng camera trực tiếp
cap = cv2.VideoCapture(0)

# Tải ảnh overlay
overlay = cv2.imread('./tutorial/cool.png', cv2.IMREAD_UNCHANGED)

# Thiết lập Face Detection với độ chính xác cao
with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5) as face_detection:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Chuyển đổi sang RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Cải thiện hiệu suất
        image_rgb.flags.writeable = False
        results = face_detection.process(image_rgb)
        
        # Vẽ kết quả Face Detection lên ảnh
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                # Lấy tọa độ bounding box của khuôn mặt
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                x, y, w, h = bbox

                # Điều chỉnh kích thước của ảnh overlay (1.5 lần kích thước bounding box)
                overlay_resize = cv2.resize(overlay, (int(w * 1.5), int(h * 1.5)))

                # Đặt ảnh overlay lên khung hình
                frame = cvzone.overlayPNG(frame, overlay_resize, [x - 45, y - 120])

        # Hiển thị khung hình
        cv2.imshow('MediaPipe Face Detection with Overlay', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
