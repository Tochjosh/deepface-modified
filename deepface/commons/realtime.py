import os
import re
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import deepface.DeepFace as DeepFace
import deepface.extendedmodels.Age as Age
import deepface.commons.functions as functions
import deepface.commons.distance as dst
import deepface.detectors.FaceDetector as FaceDetector


def analysis(db_path, model_name='VGG-Face', detector_backend='opencv', distance_metric='cosine',
             enable_face_analysis=False, source=0, time_threshold=5, frame_threshold=5):
    # ------------------------

    face_detector = FaceDetector.build_model(detector_backend)
    print("Detector backend is ", detector_backend)

    # ------------------------

    input_shape = (224, 224)
    input_shape_x = input_shape[0]
    input_shape_y = input_shape[1]

    text_color = (255, 255, 255)

    employees = []
    # check passed db folder exists
    if os.path.isdir(db_path):
        for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
            for file in f:
                if '.jpg' in file:
                    # exact_path = os.path.join(r, file)
                    exact_path = r + "/" + file
                    employees.append(exact_path)

    if len(employees) == 0:
        print("WARNING: There is no image in this path ( ", db_path, ") . Face recognition will not be performed.")

    # ------------------------
    model = None
    if len(employees) > 0:
        model = DeepFace.build_model(model_name)
        print(model_name, "is built")

        # ------------------------

        input_shape = functions.find_input_shape(model)
        input_shape_x = input_shape[0]
        input_shape_y = input_shape[1]

        # tuned thresholds for model and metric pair
        threshold = dst.findThreshold(model_name, distance_metric)

    # ------------------------
    # facial attribute analysis models

    emotion_model = None
    age_model = None
    gender_model = None

    if enable_face_analysis:
        tic = time.time()

        emotion_model = DeepFace.build_model('Emotion')
        print("Emotion model loaded")

        age_model = DeepFace.build_model('Age')
        print("Age model loaded")

        gender_model = DeepFace.build_model('Gender')
        print("Gender model loaded")

        toc = time.time()

        print("Facial attribute analysis models loaded in ", toc - tic, " seconds")

    # ------------------------

    # find embeddings for employee list

    tic = time.time()

    # -----------------------

    pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')

    # TODO: why don't you store those embeddings in a pickle file similar to find function?

    embeddings = []
    # for employee in employees:
    for index in pbar:
        employee = employees[index]
        pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
        embedding = []

        # preprocess_face returns single face. this is expected for source images in db.
        img = functions.preprocess_face(img=employee, target_size=(input_shape_y, input_shape_x),
                                        enforce_detection=False, detector_backend=detector_backend)
        img_representation = model.predict(img)[0, :]

        embedding.append(employee)
        embedding.append(img_representation)
        embeddings.append(embedding)

    df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
    df['distance_metric'] = distance_metric

    toc = time.time()

    print("Embeddings found for given data set in ", toc - tic, " seconds")

    # -----------------------

    pivot_img_size = 112  # face recognition result image

    # -----------------------

    # freeze = False
    face_detected = False
    # face_included_frames = 0  # freeze screen if face detected sequentially 5 frames
    # freezed_frame = 0
    # tic = time.time()

    cap = cv2.VideoCapture(source)  # webcam

    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        if img is None:
            break

        raw_img = img.copy()
        resolution = img.shape
        resolution_x = img.shape[1]
        resolution_y = img.shape[0]

        try:
            # faces store list of detected_face and region pair
            faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align=True)
        except:  # to avoid exception if no face detected
            faces = []

        # if len(faces) == 0:
        #     face_included_frames = 0
        # # else:
        # #     faces = []

        detected_faces = []
        # face_index = 0
        for face, (x, y, w, h) in faces:
            # if w > 130:  # discard small detected faces

            face_detected = True

            cv2.rectangle(img, (x, y), (x + w, y + h), (67, 67, 67), 1)  # draw rectangle to main image

            # detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face

            # -------------------------------------

            detected_faces.append((x, y, w, h))

        # -------------------------------------

        if face_detected:

            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()

            freeze_img = base_img.copy()

            for detected_face in detected_faces_final:
                x = detected_face[0]
                y = detected_face[1]
                w = detected_face[2]
                h = detected_face[3]

                cv2.rectangle(freeze_img, (x, y), (x + w, y + h), (67, 67, 67),
                              3)  # draw rectangle to main image

                # -------------------------------

                # apply deep learning for custom_face

                custom_face = base_img[y:y + h, x:x + w]

                # -------------------------------
                # facial attribute analysis
                enable_face_analysis = False  # temporarily shut down
                if enable_face_analysis:

                    gray_img = functions.preprocess_face(img=custom_face, target_size=(48, 48), grayscale=True,
                                                         enforce_detection=False, detector_backend='opencv')
                    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                    emotion_predictions = emotion_model.predict(gray_img)[0, :]
                    sum_of_predictions = emotion_predictions.sum()

                    mood_items = []
                    for i in range(0, len(emotion_labels)):
                        mood_item = []
                        emotion_label = emotion_labels[i]
                        emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                        mood_item.append(emotion_label)
                        mood_item.append(emotion_prediction)
                        mood_items.append(mood_item)

                    emotion_df = pd.DataFrame(mood_items, columns=["emotion", "score"])
                    emotion_df = emotion_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)

                    # background of mood box

                    # transparency
                    overlay = freeze_img.copy()
                    opacity = 0.4

                    if x + w + pivot_img_size < resolution_x:
                        # right
                        cv2.rectangle(freeze_img
                                      , (x + w, y)
                                      , (x + w + pivot_img_size, y + h)
                                      , (64, 64, 64), cv2.FILLED)

                        cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                    elif x - pivot_img_size > 0:
                        # left
                        cv2.rectangle(freeze_img
                                      , (x - pivot_img_size, y)
                                      , (x, y + h)
                                      , (64, 64, 64), cv2.FILLED)

                        cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                    for index, instance in emotion_df.iterrows():
                        emotion_label = "%s " % (instance['emotion'])
                        emotion_score = instance['score'] / 100

                        bar_x = 35  # this is the size if an emotion is 100%
                        bar_x = int(bar_x * emotion_score)

                        if x + w + pivot_img_size < resolution_x:

                            text_location_y = y + 20 + (index + 1) * 20
                            text_location_x = x + w

                            # if text_location_y < y + h:
                            #     cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y),
                            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            #
                            #     cv2.rectangle(freeze_img
                            #                   , (x + w + 70, y + 13 + (index + 1) * 20)
                            #                   , (x + w + 70 + bar_x, y + 13 + (index + 1) * 20 + 5)
                            #                   , (255, 255, 255), cv2.FILLED)

                        elif x - pivot_img_size > 0:

                            text_location_y = y + 20 + (index + 1) * 20
                            text_location_x = x - pivot_img_size

                            if text_location_y <= y + h:
                                cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                                cv2.rectangle(freeze_img
                                              , (x - pivot_img_size + 70, y + 13 + (index + 1) * 20)
                                              , (x - pivot_img_size + 70 + bar_x, y + 13 + (index + 1) * 20 + 5)
                                              , (255, 255, 255), cv2.FILLED)

                    # -------------------------------

                    face_224 = functions.preprocess_face(img=custom_face, target_size=(224, 224),
                                                         grayscale=False, enforce_detection=False,
                                                         detector_backend='opencv')

                    age_predictions = age_model.predict(face_224)[0, :]
                    apparent_age = Age.findApparentAge(age_predictions)

                    # -------------------------------

                    gender_prediction = gender_model.predict(face_224)[0, :]

                    gender = None
                    if np.argmax(gender_prediction) == 0:
                        gender = "W"
                    elif np.argmax(gender_prediction) == 1:
                        gender = "M"

                    analysis_report = str(int(apparent_age)) + " " + gender

                    # -------------------------------

                    info_box_color = (46, 200, 255)

                    # top
                    if y - pivot_img_size + int(pivot_img_size / 5) > 0:

                        triangle_coordinates = np.array([
                            (x + int(w / 2), y)
                            , (x + int(w / 2) - int(w / 10), y - int(pivot_img_size / 3))
                            , (x + int(w / 2) + int(w / 10), y - int(pivot_img_size / 3))
                        ])

                        cv2.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)

                        cv2.rectangle(freeze_img,
                                      (x + int(w / 5), y - pivot_img_size + int(pivot_img_size / 5)),
                                      (x + w - int(w / 5), y - int(pivot_img_size / 3)), info_box_color,
                                      cv2.FILLED)

                        cv2.putText(freeze_img, analysis_report,
                                    (x + int(w / 3.5), y - int(pivot_img_size / 2.1)), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 111, 255), 2)

                    # bottom
                    elif y + h + pivot_img_size - int(pivot_img_size / 5) < resolution_y:

                        triangle_coordinates = np.array([
                            (x + int(w / 2), y + h)
                            , (x + int(w / 2) - int(w / 10), y + h + int(pivot_img_size / 3))
                            , (x + int(w / 2) + int(w / 10), y + h + int(pivot_img_size / 3))
                        ])

                        cv2.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)

                        cv2.rectangle(freeze_img, (x + int(w / 5), y + h + int(pivot_img_size / 3)),
                                      (x + w - int(w / 5), y + h + pivot_img_size - int(pivot_img_size / 5)),
                                      info_box_color, cv2.FILLED)

                        cv2.putText(freeze_img, analysis_report,
                                    (x + int(w / 3.5), y + h + int(pivot_img_size / 1.5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

                # -------------------------------
                # face recognition

                custom_face = functions.preprocess_face(img=custom_face,
                                                        target_size=(input_shape_y, input_shape_x),
                                                        enforce_detection=False, detector_backend='opencv')

                # check preprocess_face function handled
                if custom_face.shape[1:3] == input_shape:
                    if df.shape[0] > 0:  # if there are images to verify, apply face recognition
                        img1_representation = model.predict(custom_face)[0, :]

                        def find_distance(row):
                            distance_metric = row['distance_metric']
                            img2_representation = row['embedding']

                            distance = 1000  # initialize very large value
                            if distance_metric == 'cosine':
                                distance = dst.findCosineDistance(img1_representation, img2_representation)
                            elif distance_metric == 'euclidean':
                                distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                            elif distance_metric == 'euclidean_l2':
                                distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation),
                                                                     dst.l2_normalize(img2_representation))

                            return distance

                        df['distance'] = df.apply(find_distance, axis=1)
                        df = df.sort_values(by=["distance"])

                        candidate = df.iloc[0]
                        best_distance = candidate['distance']
                        employee_name = candidate['employee']

                        if best_distance <= threshold:
                            acc = round(best_distance - 1) * 10

                            label = employee_name.split("/")[-1].replace(".jpg", "")
                            label = re.sub('[0-9]', '', label)

                            try:
                                if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                                    # top right
                                    # freeze_img[y - pivot_img_size:y, x + w:x + w + pivot_img_size] = display_img

                                    overlay = freeze_img.copy()
                                    opacity = 0.4
                                    cv2.rectangle(freeze_img, (x + w, y), (x + w + pivot_img_size, y + 20),
                                                  (0, 0, 255), cv2.FILLED)
                                    # cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                    cv2.putText(freeze_img, f"{label} {acc}%", (x + w, y + 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, text_color, 2)

                                    # connect face and text
                                    # cv2.line(freeze_img, (x + int(w / 2), y),
                                    #          (x + 3 * int(w / 4), y - int(pivot_img_size / 2)), (67, 67, 67), 1)
                                    # cv2.line(freeze_img, (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                    #          (x + w, y - int(pivot_img_size / 2)), (67, 67, 67), 1)

                                elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                                    # bottom left
                                    # freeze_img[y + h:y + h + pivot_img_size, x - pivot_img_size:x] = display_img

                                    overlay = freeze_img.copy()
                                    opacity = 0.4
                                    cv2.rectangle(freeze_img, (x - pivot_img_size, y + h - 20), (x, y + h),
                                                  (0, 0, 255), cv2.FILLED)
                                    # cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                    cv2.putText(freeze_img, f"{label} {acc}%",
                                                (x - pivot_img_size, y + h - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                                    # connect face and text
                                    # cv2.line(freeze_img, (x + int(w / 2), y + h),
                                    #          (x + int(w / 2) - int(w / 4), y + h + int(pivot_img_size / 2)),
                                    #          (67, 67, 67), 1)
                                    # cv2.line(freeze_img,
                                    #          (x + int(w / 2) - int(w / 4), y + h + int(pivot_img_size / 2)),
                                    #          (x, y + h + int(pivot_img_size / 2)), (67, 67, 67), 1)

                                elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                                    # top left
                                    # freeze_img[y - pivot_img_size:y, x - pivot_img_size:x] = display_img

                                    overlay = freeze_img.copy()
                                    opacity = 0.4
                                    cv2.rectangle(freeze_img, (x - pivot_img_size, y), (x, y + 20),
                                                  (0, 0, 255), cv2.FILLED)
                                    # cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                    cv2.putText(freeze_img, f"{label} {acc}%",
                                                (x - pivot_img_size, y + 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                                    # connect face and text
                                    # cv2.line(freeze_img, (x + int(w / 2), y),
                                    #          (x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                                    #          (67, 67, 67), 1)
                                    # cv2.line(freeze_img,
                                    #          (x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                                    #          (x, y - int(pivot_img_size / 2)), (67, 67, 67), 1)

                                elif x + w + pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                                    # bottom right
                                    # freeze_img[y + h:y + h + pivot_img_size,
                                    # x + w:x + w + pivot_img_size] = display_img

                                    overlay = freeze_img.copy()
                                    opacity = 0.4
                                    cv2.rectangle(freeze_img, (x + w, y + h - 20),
                                                  (x + w + pivot_img_size, y + h), (0, 0, 255), cv2.FILLED)
                                    # cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                    cv2.putText(freeze_img, f"{label} {acc}%",
                                                (x + w, y + h - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                                    # connect face and text
                                    # cv2.line(freeze_img, (x + int(w / 2), y + h),
                                    #          (x + int(w / 2) + int(w / 4), y + h + int(pivot_img_size / 2)),
                                    #          (67, 67, 67), 1)
                                    # cv2.line(freeze_img,
                                    #          (x + int(w / 2) + int(w / 4), y + h + int(pivot_img_size / 2)),
                                    #          (x + w, y + h + int(pivot_img_size / 2)), (67, 67, 67), 1)

                            except Exception as err:
                                print(str(err))

            cv2.imshow('img', freeze_img)

        else:
            cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
