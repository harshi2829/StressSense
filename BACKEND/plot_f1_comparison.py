import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]

# Approximate F1 values from your runs
f1_cnn_bilstm = [0.47, 0.43, 0.47, 0.44, 0.50]          # no weights, no attention
f1_weighted_bilstm = [0.56, 0.60, 0.61, 0.61, 0.62]     # class weights, no attention
f1_weighted_att = [0.68, 0.73, 0.75, 0.79, 0.79]        # class weights + attention

plt.figure(figsize=(6, 4))
plt.plot(epochs, f1_cnn_bilstm, marker='o', label='CNN-BiLSTM')
plt.plot(epochs, f1_weighted_bilstm, marker='o', label='CNN-BiLSTM + weights')
plt.plot(epochs, f1_weighted_att, marker='o', label='CNN-BiLSTM + weights + attention')

plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.title('F1 over epochs for different models')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
