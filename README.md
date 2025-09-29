# 🖼️ Pix2Caption – AI Image Caption Generator  
*“Let AI describe what it sees.”* 🤖📝  
Pix2Caption is an **image-to-text** application that analyzes a photo and generates a natural-language caption in real time. Built with 🧠 TensorFlow/Keras and 🎨 Streamlit, it showcases the fusion of **Computer Vision** and **Natural Language Processing**.

---

## 🧠 Problem Statement  
Manually describing large numbers of images is tedious and inconsistent.  
> ❓ *“Can we automatically create meaningful captions for any image?”*  
> ✅ *Yes—Pix2Caption proves it with a deep learning encoder–decoder model.*

---

## 🌐 Live Demo  
Visit the deployed Streamlit app to upload any image and get an instant AI-generated caption.


https://pix2caption-image-captioning.streamlit.app/

---

## 📦 Features  
- 📤 **Image Upload** – Supports JPG, JPEG, PNG  
- 🧠 **Deep Learning Model** – DenseNet-201 CNN encoder + LSTM decoder  
- 🔤 **Tokenizer & Sequence Handling** – Start/End tokens, padded sequences  
- ⚡ **Real-Time Inference** – Generates captions on-the-fly  
- 💾 **Download** – Save generated captions as `.txt` files  
- 🎨 **Responsive UI** – Modern Streamlit interface with custom CSS

---

## 🏗️ Architecture  

```text
User Image ➡️ Feature Extractor (DenseNet-201 CNN)
             ➡️ LSTM Decoder with Tokenizer
             ➡️ Word-by-word Caption Generation
```
---

## Project files
| File/Folder                      | Description                                            |
| -------------------------------- | ------------------------------------------------------ |
| `Pix2Caption.ipynb`              | Model training, preprocessing, and evaluation notebook |
| `main.py`                        | Streamlit frontend for deployment                      |
| `models/caption_model.keras`     | Trained LSTM decoder                                   |
| `models/feature_extractor.keras` | CNN encoder for image feature extraction               |
| `models/tokenizer.pkl`           | Saved tokenizer vocabulary                             |
| `requirements.txt`               | Python dependencies                                    |


--- 
## 🔐 Technologies Used

- 🧠 TensorFlow / Keras – Model building & training

- 🖼️ DenseNet-201 – Pretrained CNN encoder

- 🔁 LSTM – Text sequence decoder

- 📦 Streamlit – Interactive web UI

- 🐍 Python – NumPy, Pillow, Pickle, etc.
---

## 🚀 Installation & Setup
1️⃣ Clone the repo
```bash
git clone https://github.com/Manthan2110/Pix2Caption.git
cd Pix2Caption
```

2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Run the Streamlit app
```bash
streamlit run main.py
```

---

## 📈 How It Works

Upload an image from your device.

Encoder (DenseNet-201) extracts visual feature vectors.

Decoder (LSTM) predicts the next word step-by-step until the <end> token.

The generated caption is displayed and available for download.

---

## Future Enchantment
| Feature                         | Description                                      |
| ------------------------------- | ------------------------------------------------ |
| 🔍 Attention Mechanism          | Improve caption relevance with spatial attention |
| 🌐 Multi-language Support       | Generate captions in different languages         |
| 🧠 Transformer Decoder          | Upgrade to a Transformer for richer descriptions |
| 🌌 Beam Search / Top-k Sampling | Produce more diverse, higher-quality captions    |
| 📊 Evaluation Metrics           | Integrate BLEU, METEOR, CIDEr scoring            |

---
## 👨‍💻 Author

Made with ❤️ by Manthan Jadav

--- 
## 📜 License

This project is licensed under the MIT License.
Feel free to fork, modify, and share!
