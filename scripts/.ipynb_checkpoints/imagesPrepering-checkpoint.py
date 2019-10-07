import cv2
from time import sleep
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
from PIL import Image

class ImagesForModel:
    """
    Klasa 'ImagesForModel' odpowiada za dostep do kamery uzytkownika, przechwytywanie obrazu z urzadzenia
    oraz za zapisanie przechwyconych ramek.
    Klasa ta dodatkowo umozliwia przetworzenie zapisanych ramek na macierze liczbowe.
    """
    def __init__(self):
        pass
    
    def captureFrame(self, path_for_save,number_of_frame_to_capture = 100, sleep_time = 2, number_of_device = 0):
        """
        Funckja 'captureFrame' służy do pobierania obrazu video z karery urządzenia o numerze 'number_of_device',
        obraz wideo otrzymywany z urządzenia zapisywany jest w formacie .jpg.
        Kolejne ramki pobierane sa co 'sleep_time'.
        Ilość ramek pobieranych okreslamy parametrem 'number_of_frame_to_capture'.
        Plki ze zdjeciami zapisywane sa w folderze definiowanym parametrem 'path_to_save'.
        """ 
        
        # ilosc plikow o nazwie frame* w podanym przez uzytkownika folderze.
        count = 0
        for root, dirs, files in os.walk(path_for_save+"/"):  
            for filename in files:
                if filename[:5] == "frame":
                    count += 1
        recived_images = 0
        # Przechwytywanie i zapisywanie obrazu, pliki beda zpisane pod nazwa frame*.jpg
        vidcap = cv2.VideoCapture(number_of_device)
        vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT,30)
        vidcap.set(cv2.CAP_PROP_FRAME_WIDTH,90)
        success,image = vidcap.read()
        while recived_images < number_of_frame_to_capture:
            success,image = vidcap.read()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(path_for_save+"/frame%d.jpg" % count, gray_image)     # save frame as JPEG file
            count += 1
            recived_images += 1
            sleep(2)
            
        vidcap.release() # Zamkniecie kamery
        
    def convertIntoArray(self, path_to_folder):
        """
        Funckcja zwrca zdjecia z zadanego folderu w formie macierzy liczbowej z 3 kanalami kolorowymi.
        Zdjecia zwrocone sa w jednej macierzy numpy.array
        """
        array = []
        for root, dirs, files in os.walk(path_to_folder+"/"):
            for filename in files:
                file = np.array(cv2.imread(path_to_folder+"/"+filename))
                if file.shape != ():
                    file.resize(file.shape[0], file.shape[1])
                    array.append(file)
                
        return np.array(array)
    
    def changeImageToGreyScale(self, path_to_image):
        """
        Funkcja podmienia zdjecie w podanej lokalizacji na to samo zdjecie tylko z kolorami w skali szarosci.
        """
        image = cv2.imread(path_to_image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.imwrite(path_to_image, gray_image)

    
    def resizeImage(self, path_to_image, height, width, save = True):
        """
        Funkcja zmienia rozmiar zdjecia w podanej lokalizacji nastepnie nadpisuje plik z grafika.
        """
        image = cv2.imread(path_to_image)
        resized_image = cv2.resize(image, (width, height))
        if save:
            return cv2.imwrite(path_to_image, resized_image)
        else:
            return np.array(resized_image)
    
    def createSquare(self):
        """
        Funckja tworzy plik graficzny który zawiere reprezentacje klasy 'Square',
        Czyli biały prosotokąt (same linie) na czarnym tle.
        Plik graficzny ma rozmiar 200x200.
        """
        square_class_image = np.zeros((200,200,3))
        color = (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255))
        size = 100
        start_point = int((square_class_image.shape[0] - size) / 2)
        end_point = int(square_class_image.shape[0] - start_point)
        square_class_image += self.odcinek(200,200, (start_point, start_point), (end_point, start_point), 10, color)
        square_class_image += self.odcinek(200,200, (start_point, start_point), (start_point, end_point), 10, color)
        square_class_image += self.odcinek(200,200, (start_point, end_point), (end_point, end_point), 10, color)
        square_class_image += self.odcinek(200,200, (end_point, start_point), (end_point, end_point), 10, color)
        try:
            os.mkdir("../class_object_image")
            os.remove("../class_object_image/square.jpg")
        except:
            pass
        cv2.imwrite("../class_object_image/square.jpg", square_class_image)
    
    def createTriangle(self):
        """
        Funckja tworzy plik graficzny który zawiere reprezentacje klasy 'Triangle',
        Czyli biały trójkąt (same linie) na czarnym tle.
        Plik graficzny ma rozmiar 200x200.
        """
        triangle_class_image = np.zeros((200,200,3))
        color = (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255))
        triangle_class_image += self.odcinek(200,200, (143, 50), (143, 150), 10, color)
        triangle_class_image += self.odcinek(200,200, (143, 50), (56, 100), 10, color)
        triangle_class_image += self.odcinek(200,200, (56, 100), (143, 150), 10, color)
        try:
            os.mkdir("../class_object_image")
            os.remove("../class_object_image/triangle.jpg")
        except:
            pass
        cv2.imwrite("../class_object_image/triangle.jpg", triangle_class_image.T.T)
        
    def rotate(self, image, angle=90, scale=1.0):
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image
    
    def placeRandom(self, path_to_class_object, path_to_image, rotate=False):
        """
        Funckja ta modyfikuje wejsciowy plik graficzny dodajac do niego obiekt
        który bedzie wyszukiwany przez sieć neuronową.
        Grafika z dodawana klasą jest zmiejszana do losowego rozmiaru z zakresu od 40x40 do 90x90
        Następnie jest dodawany na początkowa grafike w losowym miejscu.
        Plik początkowy następnie jest nadpisywany.
        """
        image = cv2.imread(path_to_image)
        new_size_of_object = random.randrange(40, 90)
        custom_object = self.resizeImage(path_to_class_object, new_size_of_object, new_size_of_object, False)
        empty_board = np.zeros(image.shape)
        pos_x = random.randrange(image.shape[0] - new_size_of_object)
        pos_y = random.randrange(image.shape[1] - new_size_of_object)
        for i in range(0, new_size_of_object):
            for j in range(0, new_size_of_object):
                if sum(custom_object[i][j]) >= 10:
                    image[pos_x+i][pos_y+j] = custom_object[i][j]
        if rotate == True:
            rota = random.randint(0,180)
            image = self.rotate(image, rota)
        cv2.imwrite(path_to_image, image)
    
    def resizeImagesInFolder(self, path_to_folder, width, height):
        for root, dirs, files in os.walk(path_to_folder+"/"):
            for directory in dirs:
                if (directory[0] == "."):
                    continue
                for root2, dirs2, files2 in os.walk(path_to_folder+"/"+directory+"/"):
                    for filename in files2:
                        if filename[-3:] == "jpg" and "checkpoint" not in filename:
                            self.resizeImage(path_to_folder+"/"+directory+"/"+filename, width, height)
            for filename in files:
                if filename[-3:] == "jpg" and "checkpoint" not in filename:
                    self.resizeImage(path_to_folder+"/"+filename, width, height)
                
    def addObjectImagesInFolder(self, path_to_folder, object_name = "square", rotate=False):
        for root, dirs, files in os.walk(path_to_folder+"/"):
            for filename in files:
                if filename[-3:] == "jpg" and "checkpoint" not in filename:
                    image = path_to_folder+"/"+filename
                    if object_name == "square":
                        self.createSquare()
                        self.placeRandom("../class_object_image/square.jpg", image, rotate)
                    elif object_name == "triangle":
                        self.createTriangle()
                        self.placeRandom("../class_object_image/triangle.jpg", image, rotate)
    
    def greyscaleImagesInFolder(self, path_to_folder):
        for root, dirs, files in os.walk(path_to_folder+"/"):
            for filename in files:
                if filename[-3:] == "jpg" and "checkpoint" not in filename:
                    self.changeImageToGreyScale(path_to_folder+"/"+filename, )
                    
    def copyFilesToAnotherFolder(self, path_to_orgianals, path_for_copies, name_format_for_copies):
        iter_for_name = 0
        for root, dirs, files in os.walk(path_to_orgianals+"/"):
            for directory in dirs:
                for root, dirs, files in os.walk(path_to_orgianals+"/"+directory+"/"):
                    for filename in files:
                        if filename[-3:] == "jpg" and "checkpoint" not in filename:
                            image = cv2.imread(path_to_orgianals+"/"+directory+"/"+filename)
                            if np.array(image,dtype=np.float64).sum() > 1:
                                iter_for_name += 1
                                cv2.imwrite(path_for_copies+"/"+name_format_for_copies+str(iter_for_name)+".jpg", np.array(image,dtype=np.float64))
            for filename in files:
                if filename[-3:] == "jpg" and "checkpoint" not in filename:
                    image = cv2.imread(path_to_orgianals+"/"+filename)
                    if np.array(image,dtype=np.float64).sum() > 1:
                        iter_for_name += 1
                        cv2.imwrite(path_for_copies+"/"+name_format_for_copies+str(iter_for_name)+".jpg", np.array(image,dtype=np.float64))

    def odcinek(self, m, n, p1, p2, grubosc = 1, color = (255,255,255)):
    # Utworzenie tablicy o rozmiarze M x N wypełnioną zerami (kolor czarny)
        data = np.zeros((m,n,3))
    
    # Obliczenie przyrostu X i Y
        deltaX = p2[0] - p1[0]
        deltaY = p2[1] - p1[1]
    
    # Przypadek w ktorym punkty p1 i p2 są takaie same wieć rysujemy tylko jedne punkt dowolnie p1 lub p2
        if deltaX == 0 and deltaY == 0:
        # Uzupelniamy tablice w podanym punktie o kolor w tym przypadku [1.0, 1.0, 0.0] to kolor żółty
            data[p1[0]][p1[1]] = color
    
    # Sprawdzamy który przyrost jest większy, X czy Y i wzależności od tego iterujemy albo po X albo po Y
        elif math.fabs(deltaY) <= math.fabs(deltaX):
        # jeżeli puntk konocwy ma mniejsza wartosc X to zamieniamy punkty miejsacami ze soba
            if p1[0] > p2[0]:
                p1, p2 = p2, p1
        # ponownie obliczamy przyrost X i Y
            deltaX = p2[0] - p1[0]
            deltaY = p2[1] - p1[1]
        
            y = p1[1]
            yi = 1
            if deltaY < 0:
                yi = - 1
                deltaY = - deltaY
            d = 2*deltaY - deltaX
            for i in range(p1[0], p2[0]+1):
                if d > 0:
                    y = y + yi
                    d = d - 2*deltaX
                d = d + 2*deltaY
                if grubosc < 2:
                    data[i][y] = color
                else:
                    for new_y in range(y - int(grubosc/2), y + int(grubosc/2)):
                        data[i][new_y]= color
        else:
        # jeżeli puntk konocwy ma mniejsza wartosc 2 to zamieniamy punkty miejsacami ze soba
            if p1[1] > p2[1]:
                p1, p2 = p2, p1
        # ponownie obliczamy przrost X i y
            deltaX = p2[0] - p1[0]
            deltaY = p2[1] - p1[1]
        
            x = p1[0]
            xi = 1
            if deltaX < 0:
                xi = -1
                deltaX = -deltaX
            d = 2*deltaX - deltaY
            for i in range(p1[1], p2[1]+1):
                if d > 0:
                    x = x + xi
                    d = d- 2*deltaY
                d = d + 2*deltaX
                if grubosc < 2:
                    data[x][i]= color
                else:
                    for new_x in range(x - int(grubosc/2), x + int(grubosc/2)):
                        data[new_x][i]= color
        return data