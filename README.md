# **BERT COVID Tweet Classification**  
📢 **A deep learning project using BERT to classify COVID-19-related tweets with high accuracy.**  

---

## 📜 **Project Overview**  
This repository contains an **NLP-based classification system** that analyzes **COVID-19-related tweets** to determine whether they contain relevant pandemic-related information.  
We fine-tune **BERT (Bidirectional Encoder Representations from Transformers)** to enhance text classification performance using **TensorFlow and Hugging Face Transformers**.

---

## ✨ **Key Features**  
🔹 **Fine-tuned BERT Model** – Adapted for classifying COVID-related tweets with high accuracy.  
🔹 **Robust Evaluation Metrics** – Accuracy, Precision, Recall, and F1-score used for performance validation.  
🔹 **Preprocessing Pipeline** – Includes text cleaning, tokenization, and handling imbalanced data.  
🔹 **Visualization** – Confusion matrices and performance graphs for insight into model effectiveness.  
🔹 **Automated Training & Prediction Pipeline** – Supports real-time tweet classification.  

---

## 📊 **Results & Performance**  
The fine-tuned **BERT model** delivered the following performance:  

- **Accuracy:** **94.18%**  
- **F1-Score:** **85.73**  
- **Precision:** **98.1%**  
- **Recall:** **99.0%**  

📈 **Performance Metrics Visualizations**:  
- Confusion Matrix for Model Evaluation  
- Test Accuracy over Training Epochs  

---

## 🛠️ **Methodology**  

### **1️⃣ Data Collection & Preprocessing**  
✅ Collected **COVID-19-related tweets** from the CDC, CNN, and other verified sources.  
✅ Applied **text cleaning techniques**:  
   - Removed **retweets (RT)**, URLs, HTML entities, numbers, and special characters.  
   - Converted all text to **lowercase** for uniformity.  
✅ Tokenized text using **BERT Tokenizer** for efficient model training.  

### **2️⃣ Model Training**  
✅ Fine-tuned **BERT (bert-base-uncased)** on **cleaned and labeled tweet data**.  
✅ Splitted dataset into **70% training and 30% validation** for better generalization.  
✅ Optimized the model using:  
   - **AdamW Optimizer**  
   - **Learning rate scheduling**  

### **3️⃣ Model Evaluation & Testing**  
✅ Evaluated performance using **accuracy, precision, recall, and F1-score**.  
✅ Visualized **Confusion Matrix** to analyze classification errors.  

### **4️⃣ Prediction on New Data**  
✅ Classified **unlabeled tweets** from various sources using the trained model.  
✅ Generated **automated reports** based on predictions.  

---

## 📂 **Repository Structure**  

📁 `data/` – Contains labeled and unlabeled tweet datasets.  
📁 `notebooks/` – Jupyter Notebooks for training and evaluation.  
📁 `scripts/` – Python scripts for preprocessing, training, and prediction.  
📁 `results/` – Performance metrics, confusion matrix, and accuracy graphs.  

---

## 🔧 **Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yourusername/BERT-COVID-Tweet-Classification.git
cd BERT-COVID-Tweet-Classification
```

### **2️⃣ Install Required Libraries**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Training Script**  
```bash
python scripts/train_model.py
```

### **4️⃣ Run Prediction on New Data**  
```bash
python scripts/predict.py --input data/new_tweets.csv
```

---

## 📌 **Example Usage**  

### **Training the Model**
```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model on dataset
model.fit(train_dataset, epochs=5, validation_data=validation_dataset)
```

### **Predicting New Tweets**
```python
input_text = "COVID-19 cases are rising again. Stay safe!"
tokens = tokenizer(input_text, return_tensors="tf", truncation=True, padding=True)
output = model.predict(tokens)
predicted_class = tf.argmax(output.logits, axis=1).numpy()[0]
print(f"Predicted Label: {predicted_class}")
```

---

## 🚀 **Future Improvements**  
✅ Expand dataset with **real-time Twitter API data** for continuous learning.  
✅ Integrate **RoBERTa & DistilBERT** for comparative analysis.  
✅ Deploy as an **API using Flask or FastAPI** for real-time tweet classification.  
✅ Extend classification to **multi-label prediction** (e.g., news, misinformation, vaccine updates).  

---

## 📖 **About Me**  
👋 Hi! I’m **Jaya Krishna**, a passionate **Data Scientist & NLP Engineer** specializing in **BERT-based text classification and Deep Learning models**.  

📌 **Let's Connect!**  
📩 **Email**: jaya2305krishna@gmail.com  
🔗 **LinkedIn**: [linkedin.com/in/jaya23krishna](https://linkedin.com/in/jaya23krishna)  
🌟 **GitHub**: [github.com/jaya23krishna](https://github.com/jaya23krishna)  

🚀 Feel free to fork, contribute, or star ⭐ this project!  

---

