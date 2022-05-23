import cv2 as cv
import os
import time

BASE_DIR = 'database'


def _save_face_portion(img_frame, current_username, count,  pause_duration=3):
    time.sleep(pause_duration)
    filename = os.path.join(BASE_DIR, current_username, f"{current_username}{count}.jpg")
    cv.imwrite(filename, img_frame)


def collect():
    username = input('Enter your unique username: ').upper()

    while username in os.listdir(BASE_DIR):
        username = input('username already exist. Try a new username: ').lower()

    file_dir = os.path.join(BASE_DIR, username)
    cam = cv.VideoCapture(0)

    time.sleep(3)

    print("Collecting Images")
    count = 1
    while cam.isOpened():
        if username not in os.listdir(BASE_DIR):
            os.mkdir(file_dir)
        is_working, frame = cam.read()
        frame = cv.flip(frame, 1)

        num_of_times = 0
        if not is_working:
            print('cam not working')
            num_of_times += 1
            continue
        if num_of_times == 10:
            break

        frame = cv.resize(frame, (0, 0), None, 0.5, 0.5)

        # store image
        # if count < 3:
        #     if count == 0:
        print("Facial image collection will start in 3 sec. Kindly face the camera")
        _save_face_portion(frame, username, count)
        # elif 3 < count < 7:
        #     if count == 4:
        #         print('slightly turn your face left')
        #     _save_face_portion(frame, username, count)
        # elif 7 < count < 11:
        #     if count == 8:
        #         print('slightly turn your face right')
        #     _save_face_portion(frame, username, count)
        # elif 11 < count < 15:
        #     if count == 12:
        #         print('slightly turn your face up')
        #     _save_face_portion(frame, username, count)
        # else:
        #     if count == 16:
        #         print('slightly turn your face down')
        #     _save_face_portion(frame, username, count)

        cv.imshow(f"Collecting {username}'s face", frame)

        if count == 5:
            break
        count += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()
    print('Process complete, and images saved')
