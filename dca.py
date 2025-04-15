import cv2
import face_recognition
import mediapipe as mp
import os
import time

# You need to upload the images to the folder "fotos_referencia" in the same directory as this script
def cargar_fotos_referencia(carpeta):
    fotos_referencia = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            ruta_imagen = os.path.join(carpeta, archivo)
            imagen = face_recognition.load_image_file(ruta_imagen)
            encoding = face_recognition.face_encodings(imagen)
            if encoding:
                fotos_referencia.append(encoding[0])
    return fotos_referencia

fotos_referencia = cargar_fotos_referencia("fotos_referencia")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video capture
# You can change the camera index if you have multiple cameras
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# You can change the pixelation duration and the frame rate
pixelado_activo = True
frame_count = 0
process_every = 1
ultima_region = None
tiempo_ultima_deteccion = time.time()
pixelado_duracion = 2

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % process_every == 0:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        results = pose.process(rgb_frame)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            h, w, _ = frame.shape

            if any(face_recognition.compare_faces(fotos_referencia, face_encoding, tolerance=0.6)) and pixelado_activo:
                if results.pose_landmarks:
                    x_min = int(min([lm.x for lm in results.pose_landmarks.landmark]) * w)
                    x_max = int(max([lm.x for lm in results.pose_landmarks.landmark]) * w)
                    y_min = int(min([lm.y for lm in results.pose_landmarks.landmark]) * h)
                    y_max = int(max([lm.y for lm in results.pose_landmarks.landmark]) * h)

                    x_min = min(x_min, left)
                    x_max = max(x_max, right)
                    y_min = min(y_min, top)
                    y_max = max(y_max, bottom)

                    y_min = max(0, y_min - int((bottom - top) * 0.5))

                    ultima_region = (x_min, y_min, x_max, y_max)
                    tiempo_ultima_deteccion = time.time()

    tiempo_actual = time.time()
    if ultima_region and (tiempo_actual - tiempo_ultima_deteccion <= pixelado_duracion):
        x_min, y_min, x_max, y_max = ultima_region

        x_min, x_max = max(0, x_min), min(frame.shape[1], x_max)
        y_min, y_max = max(0, y_min), min(frame.shape[0], y_max)

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_LINEAR)
            roi = cv2.resize(roi, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
            frame[y_min:y_max, x_min:x_max] = roi

    cv2.imshow("Video", frame)

    # Check for key presses
    # Press 'q' to quit
    # Press 'p' to toggle pixelation
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        pixelado_activo = not pixelado_activo
        print(f"Pixelado {'activado' if pixelado_activo else 'desactivado'}.")

video_capture.release()
cv2.destroyAllWindows()
