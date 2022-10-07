# Urban Sound Classification
### Deep Learning (Convolutional Neural Network)

![](https://static.wikia.nocookie.net/bleach/images/1/16/Ep329UraharaProfileOption4.png/revision/latest/scale-to-width-down/1000?cb=20220325000742&path-prefix=en)

## Libraries
- Numpy 
- Matplotlib 
- Pandas 
- Librosa 
- Tensorflow.keras
- sklearn
## Dataset Analysis
This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes:  
| 0 = airconditioner  | 5 = engineidling |
|---------------------|------------------|
| 1 = carhorn         | 6 = gunshot      |
| 2 = childrenplaying | 7 = jackhammer   |
| 3 = dogbark         | 8 = siren        |
| 4 = drilling        | 9 = street_music |


We've used librosa to generate grayscale heatmap of the data.  
![](https://i.ibb.co/0hQ45Mq/indir.png)
## Spectrogram
We've generated spectrogram for each data with librosa.  
~~~~
y, sr = librosa.load("/content/UrbanSound/UrbanSound8K/audio/fold5/100263-2-0-36.wav")
plt.figure(figsize = (20,10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref = np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis = "linear")
plt.colorbar(format='%+2.0f dB')
~~~~
![](https://i.ibb.co/0XGy0fn/spec.png)
## Preprocessing
## Convolutional Neural Network Implementation
## Results
A Convolutional Neural Network with dropouts is used.  
The following results are obtained by using 0.33 test to train ratio.  
  
Train Accuracy: 95.90%  
Test Accuracy: 73.11%  
Train Loss: 66.11%  
Test Loss: 66.11%

## References
1. [Global AI Hub](https://globalaihub.com/courses/introduction-to-deep-learning/)
2. [Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
---
