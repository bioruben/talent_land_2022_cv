# talent_land_2022_cv

# Computer Vision y Realidad Aumentada - Talent Land 2022

Material utilizado en en el workshop de Talent Land el 20/07/2022.

## Instalar dependencias necesarias

Las versiones de las dependencias utilizadas cuando se realizaron estos demos son:

- Python 3.7.3
- OpenCV 4.6.0
- Para usar los marcadores es necesario tener la versión de contrib de OpenCV:

``` pip install opencv-contrib-python ```


## Codes

- generate_aruco.py

	Permite generar los marcadores para su impresión de acuerdo a su código.

- detect_aruco.py

	Permite detecta los marcadores y que tiene un flag para detectar los pokemones.

- matching_video.py

	Ejemplo que muestra el matching de descriptores.

- trainning_cnn.py

	Código de una CNN básica para entrenarla desde cero usando Keras.

- test_cnn.py

	Código que prueba la CNN entrenada en el paso anterior.

La útlima actualización se hizo el 10/07/2022, se actualizó el README.

## Usage

Todos los códigos tienen un menú de help para poderlo correr, ejemplo:

```python3 test_cnn.py -h```

Si se desea probar la CNN después de haber bajado los pesos del modelo o haberlo entrenado cada usuario.

```python3 test_cnn.py -i /path/to/the/image/folder```