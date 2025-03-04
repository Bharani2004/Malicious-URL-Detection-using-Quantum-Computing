import numpy as np
from qiskit import QuantumCircuit

def quantum_encoder(url):
    length = len(url)
    special_characters = sum(not c.isalnum() for c in url)
    frequency_keywords = sum(1 for keyword in ['login', 'secure'] if keyword in url) + (1 if 'https' in url else 0)
    encoded_state = np.array([length, special_characters, frequency_keywords], dtype=float)

    
    norm = np.linalg.norm(encoded_state)
    if norm == 0:
        raise ValueError("Encoded state cannot be a zero vector.")
    
    encoded_state /= norm  

    
    power_of_2_length = 2 ** np.ceil(np.log2(len(encoded_state))).astype(int)
    padded_state = np.zeros(power_of_2_length)
    padded_state[:len(encoded_state)] = encoded_state
    
    return padded_state

def QITLayer(phi, weights):
    num_qubits = int(np.log2(len(phi))) 
    qc = QuantumCircuit(num_qubits)

    print(f"Initializing QITLayer with phi: {phi}, num_qubits: {num_qubits}")
    qc.initialize(phi, [i for i in range(num_qubits)]) 

    for i in range(num_qubits):
        qc.rx(weights[i], i)
    return qc

def QNNLayer(phi, weights, bias):
    num_qubits = int(np.log2(len(phi)))  
    qc = QuantumCircuit(num_qubits)

    print(f"Initializing QNNLayer with phi: {phi}, num_qubits: {num_qubits}")
    qc.initialize(phi, [i for i in range(num_qubits)])  

    for i in range(num_qubits):
        qc.ry(weights[i], i)
    qc.measure_all()

    return qc

def quantum_loss_function(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def quantum_classifier(circuit, shots=1024):
    counts = {'0': 0, '1': 0}

    for _ in range(shots):
        if np.random.rand() < 0.5:
            counts['0'] += 1
        else:
            counts['1'] += 1

    prob_phishing = counts['1'] / shots
    return prob_phishing

def train_quantum_model(urls, epochs=10, learning_rate=0.01):
    num_layers = 2
    weights_QIT = [np.random.rand(2) for _ in range(num_layers)] 
    weights_QNN = np.random.rand(2)  
    bias = np.random.rand(1)

    for epoch in range(epochs):
        for url in urls:
            phi = quantum_encoder(url)
            for k in range(num_layers):
                QITLayer(phi, weights_QIT[k])

            l = QNNLayer(phi, weights_QNN, bias)

            target = np.array([1])  
            prediction = quantum_classifier(l)
            loss = quantum_loss_function(np.array([prediction]), target)

            
            for k in range(num_layers):
                weights_QIT[k] -= learning_rate * np.random.rand(2) 
            weights_QNN -= learning_rate * np.random.rand(2) 
            bias -= learning_rate * np.random.rand(1)

    return weights_QIT, weights_QNN, bias

def classify_url(url, threshold=0.5):
    urls = [url]
    weights_QIT, weights_QNN, bias = train_quantum_model(urls)

    phi = quantum_encoder(url)
    l = QNNLayer(phi, weights_QNN, bias)

    P_i = quantum_classifier(l)
    return "Malicious" if P_i > threshold else "Benign"

if __name__ == "__main__":
    url = "http://portalnutricional.com/uah2/index.php"
    result = classify_url(url)
    print(f"The URL '{url}' is classified as: {result}")


#https://www.google.com
#http://pmdona.ru/sites/all/Yahoo1/Indezx.html
#http://ongelezen-voda.000webhostapp.com/inloggen.html
#http://portalnutricional.com/uah2/index.php
