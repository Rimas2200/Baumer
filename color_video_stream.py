#!/usr/bin/env python3

"""
Программа для захвата видео с камеры Baumer и его отображения в цвете.
"""

import os
import sys
import cv2
import neoapi

save_dir = 'C:/Users/User/PycharmProjects/pythonProject2/Baumer/path_to_images'

try:
    camera = neoapi.Cam()
    camera.Connect()

    isColor = True
    if camera.f.PixelFormat.GetEnumValueList().IsReadable('BGR8'):
        camera.f.PixelFormat.SetString('BGR8')
    elif camera.f.PixelFormat.GetEnumValueList().IsReadable('Mono8'):
        camera.f.PixelFormat.SetString('Mono8')
        isColor = False
    else:
        print('no supported pixelformat')
        sys.exit(0)

    camera.f.ExposureTime.Set(100000)
    camera.f.AcquisitionFrameRateEnable.value = True
    camera.f.AcquisitionFrameRate.value = 60

    # Инициализируем окно с флагом изменяемого размера
    title = 'Camera Stream'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    for cnt in range(0, 20000):
        img = camera.GetImage().GetNPArray()

        cv2.resizeWindow(title, 1920, 1080)

        cv2.imshow(title, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            filename = os.path.join(save_dir, f"photo_{cnt}.jpg")
            cv2.imwrite(filename, img)
            print(f"Saved {filename}")

    cv2.destroyAllWindows()

except (neoapi.NeoException, Exception) as exc:
    print('Error: ', exc)
    sys.exit(1)

sys.exit(0)
