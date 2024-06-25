#!/usr/bin/env python3

import sys
import cv2
import neoapi

def main():
    try:
        # Инициализация камеры
        camera = neoapi.Cam()
        camera.Connect()
        camera.f.ExposureTime.Set(10000)  # Установка времени экспозиции

        # Отображение видеопотока в изменяемом окне
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

        while True:
            img = camera.GetImage()
            if not img.IsEmpty():
                imgarray = img.GetNPArray()
                cv2.imshow('Camera Feed', imgarray)

            # Прерывание цикла по нажатию клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        camera.Disconnect()

    except (neoapi.NeoException, Exception) as exc:
        print('Error:', exc)
        sys.exit(1)

if __name__ == "__main__":
    main()