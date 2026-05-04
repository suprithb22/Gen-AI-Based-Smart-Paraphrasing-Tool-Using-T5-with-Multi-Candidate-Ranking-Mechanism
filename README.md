# 🧠 Smart Paraphrasing Tool (GenAI-based)

## 📌 Project Overview

This project presents a **Generative AI-based Smart Paraphrasing Tool** that rewrites input text while preserving its original meaning and context.
It leverages the **T5 Transformer model** along with a **multi-candidate ranking mechanism** to generate high-quality, fluent, and grammatically correct paraphrases.

---

## 🎯 Aim

* Develop an AI-based text paraphrasing system
* Preserve semantic meaning of input text
* Improve readability and grammatical correctness
* Reduce plagiarism by generating unique content
* Provide a simple and user-friendly interface

---

## 🚀 Features

* ✨ AI-powered paraphrasing using T5 model
* 🔄 Multiple candidate generation using beam search
* 🏆 Ranking based on:

  * Semantic similarity
  * Fluency
  * Readability
* 🎚 Adjustable parameters:

  * Beam Width
  * Number of Candidates
  * Output Level (Fluent, etc.)
* 🌐 Interactive web-based UI
* ⚡ Real-time paraphrase generation

---

## 🖥️ Dashboard Interface

The system provides a clean and intuitive UI:

* Input paragraph section
* Preprocessing → T5 Model → Generate → Rank pipeline
* Adjustable sliders (Beam Width, Candidates)
* Output section showing best paraphrase
* Execution time display

---

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** Flask / Django (based on your implementation)
* **Model:** T5 Transformer
* **Libraries:**

  * Hugging Face Transformers
  * PyTorch / TensorFlow
  * NLP preprocessing tools

---

## ⚙️ Methodology

### 1. Input Acquisition

User provides input paragraph via web interface.

### 2. Text Preprocessing

* Remove extra spaces & symbols
* Sentence segmentation
* Tokenization

### 3. Paraphrase Generation (T5)

* Encoder understands context
* Decoder generates rewritten text

### 4. Multi-Candidate Generation

* Beam search
* Sampling techniques
* Produces multiple variations

### 5. Ranking Mechanism

Candidates evaluated based on:

* Semantic similarity
* Fluency
* Readability

### 6. Final Output

Best candidate selected and displayed as final paraphrased text.

---

## 📊 Results & Performance

* F1 Score: **0.86 – 0.89**
* Best Dataset: **Quora (F1: 0.89)**
* Precision: up to **0.91**
* Balanced Recall and strong generalization
* Stable training with convergence observed over epochs

---

## 📂 Project Structure

```
├── app.py
├── templates/
├── static/
├── model/
├── preprocessing/
├── requirements.txt
```

---

## ⚙️ Installation & Setup

1. Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
```

2. Navigate to project folder:

```
cd your-repo-name
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application:

```
python app.py
```

Open in browser:

```
http://localhost:8000/
```

---

## 🔮 Future Improvements

* Deploy on cloud platform
* Improve ranking metrics
* Add multilingual support
* Enhance UI/UX
* Integrate plagiarism checker

---

## 👨‍💻 Authors

* **Suprit H B** (ENG23AM0075)
* **Sanyam N** (ENG23AM0069)

Dayananda Sagar University
Dept. of CSE (AI & ML)

---

## 📚 References

* Vaswani et al., *Attention Is All You Need*
* Raffel et al., *T5 Transformer*
* Hugging Face Transformers Documentation
* BERT, BART, PEGASUS research papers

---

## ⭐ Conclusion

This project demonstrates an effective application of **Generative AI in text paraphrasing**, combining deep learning with intelligent ranking to produce high-quality and meaningful rewritten text suitable for real-world applications.
