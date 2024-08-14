import cv2
import mediapipe as mp
import cvzone

# Khởi tạo Face Mesh và Face Detection của MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Khởi tạo video từ camera
cap = cv2.VideoCapture(0)  # 0 để sử dụng camera mặc định

# Tải ảnh overlay
overlay_hat = cv2.imread('./tutorial/hat.png', cv2.IMREAD_UNCHANGED)
overlay_glasses = cv2.imread('./tutorial/glasses.png', cv2.IMREAD_UNCHANGED)
overlay_beard = cv2.imread('./tutorial/beard.png', cv2.IMREAD_UNCHANGED)

# Thiết lập Face Detection và Face Mesh với độ chính xác cao
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Chuyển đổi sang RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Cải thiện hiệu suất
        image_rgb.flags.writeable = False
        detection_results = face_detection.process(image_rgb)
        face_mesh_results = face_mesh.process(image_rgb)
        
        # Vẽ kết quả Face Detection lên ảnh
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if detection_results.detections and face_mesh_results.multi_face_landmarks:
            for detection, face_landmarks in zip(detection_results.detections, face_mesh_results.multi_face_landmarks):
                # Lấy tọa độ bounding box của khuôn mặt
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox

                # Điều chỉnh kích thước của ảnh overlay
                overlay_hat_resize = cv2.resize(overlay_hat, (int(w * 1.2), int(h * 1)))
                overlay_glasses_resize = cv2.resize(overlay_glasses, (int(w * 1.2), int(h * 0.5)))
                overlay_beard_resize = cv2.resize(overlay_beard, (int(w * 1.2), int(h * 0.2)))

                # Lấy tọa độ của các landmark
                # Landmark cho mũ
                hat_landmark = face_landmarks.landmark[10]
                hat_x = int(hat_landmark.x * iw)
                hat_y = int(hat_landmark.y * ih)
                # Landmark cho kính
                glasses_landmark = face_landmarks.landmark[3]
                glasses_x = int(glasses_landmark.x * iw)
                glasses_y = int(glasses_landmark.y * ih)
                # Landmark cho ria mép
                beard_landmark = face_landmarks.landmark[164]
                beard_x = int(beard_landmark.x * iw)
                beard_y = int(beard_landmark.y * ih)

                # Đặt ảnh overlay lên khung hình tại vị trí của mũ, kính và ria mép
                frame = cvzone.overlayPNG(frame, overlay_hat_resize, 
                                          [int(hat_x - overlay_hat_resize.shape[1] // 2), 
                                           int(hat_y - overlay_hat_resize.shape[0] // 1.2)])
                frame = cvzone.overlayPNG(frame, overlay_glasses_resize, 
                                          [int(glasses_x - overlay_glasses_resize.shape[1] // 2), 
                                           int(glasses_y - overlay_glasses_resize.shape[0] // 1.5)])
                frame = cvzone.overlayPNG(frame, overlay_beard_resize, 
                                          [int(beard_x - overlay_beard_resize.shape[1] // 2), 
                                           int(beard_y - overlay_beard_resize.shape[0] // 2)])

        # Hiển thị khung hình
        cv2.imshow('MediaPipe Face Detection with Overlay', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
