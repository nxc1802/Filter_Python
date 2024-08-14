import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo Face Mesh và Drawing utils của MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Thay đổi đường dẫn đến video của bạn
video_path = './video/video_test.mp4'
cap = cv2.VideoCapture(video_path)

# Tải hình ảnh bông tai và lấy kích thước
earring = cv2.imread('./tutorial/earing.jpg', cv2.IMREAD_UNCHANGED)

# Kiểm tra xem hình ảnh có kênh alpha không
has_alpha = earring.shape[2] == 4

# Thiết lập Face Mesh với độ chính xác cao
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Chuyển đổi sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Cải thiện hiệu suất
        image.flags.writeable = False
        results = face_mesh.process(image)
        
        # Vẽ kết quả Face Mesh lên ảnh
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Lấy tọa độ của các điểm tai
                right_ear = face_landmarks.landmark[147]
                left_ear = face_landmarks.landmark[376]

                # Chuyển đổi tọa độ sang pixel
                h, w, _ = image.shape
                right_ear_pixel = (int(right_ear.x * w), int(right_ear.y * h))
                left_ear_pixel = (int(left_ear.x * w), int(left_ear.y * h))

                # Tính toán khoảng cách giữa hai điểm tai (AB)
                A = np.array(right_ear_pixel)
                B = np.array(left_ear_pixel)
                AB = np.linalg.norm(A - B)

                # Thay đổi kích thước của bông tai (5% của AB)
                scale_percent = 5  # phần trăm kích thước gốc
                width = int(AB * scale_percent / 100)
                height = int(width * earring.shape[0] / earring.shape[1])
                dim = (width, height)

                # Thay đổi kích thước
                earring_resized = cv2.resize(earring, dim, interpolation=cv2.INTER_AREA)
                earring_h, earring_w = earring_resized.shape[:2]

                # Tính toán vị trí mới của bông tai
                center = (A + B) // 2
                direction = (B - A) / np.linalg.norm(B - A)
                offset = direction * (AB * 0.)
                position = center - (earring_w // 2 + offset[0], earring_h // 2 + offset[1])

                x1, y1 = int(position[0]), int(position[1])
                x2, y2 = x1 + earring_w, y1 + earring_h

                # Kiểm tra giới hạn của hình ảnh
                if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                    if has_alpha:
                        # Nếu có kênh alpha, sử dụng nó để trộn
                        alpha_s = earring_resized[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range(0, 3):
                            image[y1:y2, x1:x2, c] = (alpha_s * earring_resized[:, :, c] +
                                                      alpha_l * image[y1:y2, x1:x2, c])
                    else:
                        # Nếu không có kênh alpha, chèn trực tiếp hình ảnh
                        image[y1:y2, x1:x2] = earring_resized

        # Hiển thị hình ảnh
        cv2.imshow('MediaPipe Face Mesh with Earrings', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
