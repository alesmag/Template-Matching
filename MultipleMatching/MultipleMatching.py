'''
@Author: Alessio Giuseppe Muggittu
@Version: 1.0
@Date: 27/05/2020
'''

# Import set
import cv2 as cv
import numpy as np


# Definizione del Main
def main():

  # Legge l'immagine principale
  img = cv.imread("image.jpg")

  # Converte l'immagine principale in scala di grigi definendo una nuova immagine
  greyImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

  # Legge l'immagine "oggetto"
  template = cv.imread("template.jpg", 0)

  # Convenzionalmente si ha un set di row e column, in questo caso vogliamo ottenere l'opposto (column e row)
  width, height = template.shape[::-1]

  # "matchTemplate()" richiede tre argomenti: una immagine, un template (oggetto interno all'immagine), un metodo di ricerca
  result = cv.matchTemplate(greyImg, template, cv.TM_CCOEFF_NORMED)

  # "threshold" indica la soglia minima
  threshold = 0.8

  # "loc" indica (in questo caso) i punti piu' bianchi nell'immagine a scala di grigi
  # poiche' ognuno di essi corrisponde al punto dell'angolo in alto a sinistra della propia immagine "oggetto"
  loc = np.where(result >= threshold)

  # Ciclo for che indica i punti presenti nella variabile "loc"
  for point in zip(*loc[::-1]):
    # Disegna un rettangolo. Richiede un'immagine, un punto di inizio, un punto di fine, colore e spessore del bordo
    cv.rectangle(img, point, (point[0] + width, point[1] + height), (0, 0, 255), 1)

  # Genera una immagine risultato e la immagazzina nella stessa directory del programma
  cv.imwrite("Result.jpg", img)

  # Mostra l'immagine risultato su schermo
  cv.imshow("Image", img)

  # Non permette alla finestra di chiudersi automaticamente, aspettando un input diretto dall'utente
  cv.waitKey(0)

# Richiamo del Main
main()