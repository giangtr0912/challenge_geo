# Soil Organic Matter prediction from spectroscopic measurements
  
## **Motivation**
* Predict Soil Organic Matter using multiple machine learning
 algorithm.

## **Requirements** 
* Python (3.8.0)
* Geemap (0.11.7)
* Rasterstats (0.16.0)
* Wget (3.2)
* Geopandas (0.10.2)
* scikit-learn (1.0.2)
* Matplotlib (3.0.2) and Seaborn (0.9.0)

## **Dataset Overview**
Dataset used for **Predict Soil Organic Matter** can be downloaded from the 
link [STENON all copyrights](https://github.com/giangtr0912/challenge_geo/blob/main/data/input/challenge_geoDS.csv).
The dataset consists of spectroscopic measurements (feature columns prefixed with nir_, the
number after the prefix is the measuring wavelength in nm) identified by a unique measurement ID
(measurement_ID). Sensor readings were performed with stenon’s FarmLab device directly in the
soil on two different fields (see column location) in the US. Coordinate pairs (see column lat_lng;
WGS84) specify the location where the soil readings were performed. Column som contains
estimates of soil organic matter content in % for a soil layer of 0-30 cm depth at the position of a
FarmLab device reading.

## **Method**
* Partial least square regression (PLS) with different regularization and loss function was
  tried.
* Ridge regression was also tested.
* For cross-validation, [train_test_split](https://scikit-learn.org/ were tested.

## **Programs**
* A program was developed using Ridge and Partial least square regression algorithms. 

### **Usage:**
```python
$ python3 ./challenge.py --help 
```

### **Example**
* Clean data
```
$ python3 ./challenge.py clean
```

* Compare with POLARIS data 
```python
$ python3 ./challenge.py polaris
```

* Train the Model: Partial least square regression (PLS) and Ridge regressor
```python
$ python3 ./challenge.py predict
```