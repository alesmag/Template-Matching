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

  # Legge l'immagine "oggetto" e la converte direttamente in scala di grigi
  template = cv.imread("template.jpg", cv.IMREAD_GRAYSCALE)

  # Convenzionalmente si ha un set di row e column, in questo caso vogliamo ottenere l'opposto (column e row)
  width, height = template.shape[::-1]

  # "matchTemplate()" richiede tre argomenti: una immagine, un template (oggetto interno all'immagine), un metodo di ricerca
  result = cv.matchTemplate(greyImg, template, cv.TM_CCOEFF_NORMED)

  # "minMaxLoc()" richiede quattro variabili di assegnazione e un argomento per la ricerca (che in questo caso e' "result")
  minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
 
  # "topLeft" indica il punto nell'angolo in alto a sinistra dell'immagine "oggetto"
  topLeft = maxLoc

  # "bottomRight" indica il punto nell'angolo in basso a destra dell'immagine "oggetto"
  bottomRight = (topLeft[0] + width, topLeft[1] + height)

  # Disegna un rettangolo. Richiede un'immagine, un punto di inizio, un punto di fine, colore e spessore del bordo
  cv.rectangle(img, topLeft, bottomRight, (0, 0, 255), 2)

  # Genera una immagine risultato e la immagazzina nella stessa directory del programma
  cv.imwrite("Result.jpg", img)

  # Mostra l'immagine risultato su schermo
  cv.imshow("Image", img)

  # Non permette alla finestra di chiudersi automaticamente, aspettando un input diretto dall'utente
  cv.waitKey(0)


# Richiamo del Main
main()