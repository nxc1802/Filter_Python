import cv2 as cv
import numpy as np
import mediapipe as mp
import cvzone
import filter

# Khởi tạo các đối tượng của Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Đọc hình ảnh overlay với alpha channel
overlay_image = cv.imread('./tutorial/sharingan_basic.png', cv.IMREAD_UNCHANGED)

# Các chỉ số điểm mắt trái và phải
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def getSize(image, face_landmarks, INDEXES):
    '''
    This function computes the width and height of the face part (e.g. eye, mouth)
    '''
    image_height, image_width, _ = image.shape
    index_list = [face_landmarks[index] for index in INDEXES]
    min_x = min([landmark[0] for landmark in index_list])
    max_x = max([landmark[0] for landmark in index_list])
    min_y = min([landmark[1] for landmark in index_list])
    max_y = max([landmark[1] for landmark in index_list])
    width = max_x - min_x
    height = max_y - min_y
    
    return width, height, (int(min_x), int(min_y))

def isOpen(image, face_landmarks, face_part, threshold=2):
    '''
    This function checks whether an eye or mouth of the person(s) is open, utilizing its facial landmarks.
    '''
    # Retrieve the height and width of the image.
    image_height, image_width, _ = image.shape
    
    # Determine the indexes based on the face part
    if face_part == 'LEFT EYE':
        INDEXES = LEFT_EYE
    elif face_part == 'RIGHT EYE':
        INDEXES = RIGHT_EYE
    else:
        return False
    
    # Get the height of the face part.
    _, height, _ = getSize(image, face_landmarks, INDEXES)
    
    # Get the height of the whole face.
    _, face_height, _ = getSize(image, face_landmarks, [i for i in range(0, 468)])  # Use all landmarks for the face height
    
    # Check if the face part is open.
    return (height/face_height)*100 > threshold

def display_video():
    # Đường dẫn đến video cần xử lý
    video_path = './video/video_test.mp4'
    cap = cv.VideoCapture(video_path)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Xử lý hình ảnh
            frame_raw = cv.flip(frame, 1)
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                # Kiểm tra xem mắt trái có mở không
                left_eye_open = isOpen(frame, landmarks, 'LEFT EYE')
                
                # Kiểm tra xem mắt phải có mở không
                right_eye_open = isOpen(frame, landmarks, 'RIGHT EYE')

                if left_eye_open:
                    # Tính toán trung tâm và bán kính của tròng đen
                    (l_cx, l_cy), l_radius = cv.minEnclosingCircle(landmarks[LEFT_IRIS])
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)

                    # Resize và chèn ảnh overlay vào tròng đen
                    overlay_l = cv.resize(overlay_image, (int(2.2 * l_radius), int(2.2 * l_radius)))

                    frame = cvzone.overlayPNG(frame, overlay_l, [int(l_cx - overlay_l.shape[1] // 2), int(l_cy - overlay_l.shape[0] // 2)])

                    # Tạo mặt nạ cho vùng mắt trái
                    mask_left_eye = np.zeros_like(frame[:, :, 0])
                    cv.fillPoly(mask_left_eye, [landmarks[LEFT_EYE]], 255)

                    # Tạo mặt nạ toàn màu trắng và vẽ vùng mắt màu đen
                    left_eye_mask = np.ones_like(frame[:, :, 0]) * 255
                    cv.fillPoly(left_eye_mask, [landmarks[LEFT_EYE]], 0)
                else:
                    final_frame = frame_raw

                if right_eye_open:
                    # Tính toán trung tâm và bán kính của tròng đen
                    (r_cx, r_cy), r_radius = cv.minEnclosingCircle(landmarks[RIGHT_IRIS])
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)

                    # Resize và chèn ảnh overlay vào tròng đen
                    overlay_r = cv.resize(overlay_image, (int(2.2 * r_radius), int(2.2 * r_radius)))

                    frame = cvzone.overlayPNG(frame, overlay_r, [int(r_cx - overlay_r.shape[1] // 2), int(r_cy - overlay_r.shape[0] // 2)])

                    # Tạo mặt nạ cho vùng mắt phải
                    mask_right_eye = np.zeros_like(frame[:, :, 0])
                    cv.fillPoly(mask_right_eye, [landmarks[RIGHT_EYE]], 255)

                    # Tạo mặt nạ toàn màu trắng và vẽ vùng mắt màu đen
                    right_eye_mask = np.ones_like(frame[:, :, 0]) * 255
                    cv.fillPoly(right_eye_mask, [landmarks[RIGHT_EYE]], 0)
                else:
                    final_frame = frame_raw

            #     # Tính toán trung tâm và bán kính của tròng đen
            #     (l_cx, l_cy), l_radius = cv.minEnclosingCircle(landmarks[LEFT_IRIS])
            #     (r_cx, r_cy), r_radius = cv.minEnclosingCircle(landmarks[RIGHT_IRIS])
            #     center_left = np.array([l_cx, l_cy], dtype=np.int32)
            #     center_right = np.array([r_cx, r_cy], dtype=np.int32)

            #     # Resize và chèn ảnh overlay vào tròng đen
            #     overlay_l = cv.resize(overlay_image, (int(2.2 * l_radius), int(2.2 * l_radius)))
            #     overlay_r = cv.resize(overlay_image, (int(2.2 * r_radius), int(2.2 * r_radius)))

            #     frame = cvzone.overlayPNG(frame, overlay_l, [int(l_cx - overlay_l.shape[1] // 2), int(l_cy - overlay_l.shape[0] // 2)])
            #     frame = cvzone.overlayPNG(frame, overlay_r, [int(r_cx - overlay_r.shape[1] // 2), int(r_cy - overlay_r.shape[0] // 2)])

            #     # Tạo mặt nạ cho vùng mắt trái và phải
            #     mask_left_eye = np.zeros_like(frame[:, :, 0])
            #     mask_right_eye = np.zeros_like(frame[:, :, 0])
            #     cv.fillPoly(mask_left_eye, [landmarks[LEFT_EYE]], 255)
            #     cv.fillPoly(mask_right_eye, [landmarks[RIGHT_EYE]], 255)

            #     # Tạo mặt nạ toàn màu trắng và vẽ vùng mắt màu đen
            #     left_eye_mask = np.ones_like(frame[:, :, 0]) * 255
            #     right_eye_mask = np.ones_like(frame[:, :, 0]) * 255
            #     cv.fillPoly(left_eye_mask, [landmarks[LEFT_EYE]], 0)
            #     cv.fillPoly(right_eye_mask, [landmarks[RIGHT_EYE]], 0)

                # Áp dụng mặt nạ cho toàn bộ vùng mắt
                mask_eyes = cv.bitwise_or(mask_left_eye, mask_right_eye)
                eyes = cv.bitwise_and(left_eye_mask, right_eye_mask)

                # Che phần ảnh bị khuất bởi mí mắt
                masked_frame = cv.bitwise_and(frame, frame, mask=mask_eyes)
                without_eyes = cv.bitwise_or(frame_raw, frame_raw, mask=eyes)
                final_frame = cv.bitwise_or(without_eyes, masked_frame)
            # else:
            #     final_frame = frame_raw

            # Hiển thị ảnh tùy thuộc vào trạng thái của filter
            if filter.glasses:
                cv.imshow('Video', final_frame)
            else:
                cv.imshow('Video', frame_raw)

            # Thoát khi nhấn phím 'q'
            if cv.waitKey(10) == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    display_video()
