import numpy as np
import librosa
import pickle
from feature_extraction import LinearityDegreeFeatures, HighPowerFrequencyFeatures, _stft
import math
from sklearn import svm
from sklearn.model_selection import GridSearchCV

def load_svm_model(model_path):
    with open(model_path, 'rb') as model_file:
        clf = pickle.load(model_file)
    return clf

def extract_features_from_audio(audio_file):
    signal, _ = librosa.load(audio_file, sr=16000)

    sig_stft = _stft(signal)
    
    nfft = 2048
    S_pow = np.sum(np.abs(sig_stft)**2/nfft, axis=1)

    W = 14
    k = int((nfft/2 + 1) / W)
    power_vec = np.zeros(k)
    for i in np.arange(k):
        power_vec[i] = np.sum(S_pow[i*W:(i+1)*W])
    power_normal = power_vec / np.sum(power_vec)

    FV_LFP = power_normal[0:48] * 100

    _, FV_LDF = LinearityDegreeFeatures(power_normal)

    FV_HPF = HighPowerFrequencyFeatures(FV_LFP, omega=0.3)
    
    def lpc_to_lpcc(lpc):
        lpcc = []
        order = lpc.size - 1
        lpcc.append(math.log(order))
        lpcc.append(lpc[1])
        for i in range(2, order+1):
            sum_1 = 0
            for j in range(1, i):
                sum_1 += j / i * lpc[i-j-1] * lpcc[j]
            c = -lpc[i-1] + sum_1
            lpcc.append(c)
        return lpcc[1:13]

    def extract_lpcc(wav_path, order):
        y, _ = librosa.load(wav_path, sr=16000)
        lpc = librosa.lpc(y, order=order)
        lpcc = np.array(lpc_to_lpcc(lpc))
        return lpcc

    FV_LPC = extract_lpcc(wav_path=audio_file, order=12)

    FV_Void = np.concatenate((FV_LDF, FV_HPF, FV_LPC, FV_LFP))

    return FV_Void

def predict_audio_class(audio_file, model):
    features = extract_features_from_audio(audio_file)
    features = features.reshape(1, -1)  
    prediction = model.predict(features)
    return prediction

if __name__ == '__main__':
    model_path = "models/svm.pkl"
    
    clf = load_svm_model(model_path)
    print("SVM model loaded!")
    
    test_audio_file = "test_bhavya_audio.wav"  

    prediction = predict_audio_class(test_audio_file, clf)

    if prediction > 0.5:
        print(f"The audio file {test_audio_file} is classified as Spoof.")
    else:
        print(f"The audio file {test_audio_file} is classified as Genuine.")

        
    test_audio_file = "nikita_voice.wav"  

    prediction = predict_audio_class(test_audio_file, clf)

    if prediction < 0.5:
        print(f"The audio file {test_audio_file} is classified as Spoof.")
    else:
        print(f"The audio file {test_audio_file} is classified as Genuine.")

    test_audio_file = "fake_parakeet_voice.wav"  

    prediction = predict_audio_class(test_audio_file, clf)

    if prediction > 0.5:
        print(f"The audio file {test_audio_file} is classified as Spoof.")
    else:
        print(f"The audio file {test_audio_file} is classified as Genuine.")
