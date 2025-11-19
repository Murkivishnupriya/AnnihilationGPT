## AmbedkarGPT – AI Intern Assignment

This project is a fully local RAG-based Question Answering system built for the AI Intern Assignment at KalpIT.  
It uses **LangChain**, **ChromaDB**, **HuggingFace embeddings**, and **Mistral 7B on Ollama**.

---

## Features

- Load and preprocess `speech.txt`
- Chunk the text using LangChain
- Generate embeddings using `all-MiniLM-L6-v2`
- Store vectors in ChromaDB
- Retrieve relevant chunks using similarity search
- Use Mistral 7B via Ollama for answering
- Completely offline and free
- Clean and readable output

---

##  Installation

### 1. Clone this repo
git clone https://github.com/Murkivishnupriya/AnnihilationGPT
cd AnnihilationGPT



### 2. Install Python requirements
pip install -r requirements.txt


### 3. Install and start Ollama
Download from:
https://ollama.com/download

Pull required model:
ollama pull mistral


---

##  Running the Project
python main.py


You will be asked:


- Press **y** the first time  
- Press **n** for future runs

Then ask any question about the speech.

---

##  Project Structure
AmbedkarGPT-Intern-Task/
│── main.py
│── speech.txt
│── README.md
│── requirements.txt
└── database/ (auto-created)


---

##  Example Questions

- Why does Ambedkar criticize social reform?
- What is the root cause of caste?
- What remedy does Ambedkar suggest?
- What role do the shastras play in caste?

---
## Example Output 
User: Your question: What does Ambedkar describe as the real enemy behind caste?
AmbedkarGPT:According to Ambedkar, the real enemy behind caste is the belief in the shastras. He believes that as long as people hold the shastras as sacred and infallible, they will never be able to get rid of caste.

User; Why does Ambedkar criticize social reformers?

AmbedkarGPT:According to the context, Ambedkar criticizes social reformers because they are merely pruning the leaves and branches of a tree (i.e., addressing symptoms rather than roots), without attacking the root cause of caste, which is the belief in the shastras. He argues that as long as people believe in the sanctity of the shastras, they will never be able to get rid of caste.

##  Conclusion

This project fulfills all requirements:

✔ LangChain  
✔ ChromaDB  
✔ Mistral 7B  
✔ RAG pipeline  
✔ Clean output  
✔ GitHub ready  
✔ Fully offline  
✔ Professional documentation  

---

#  You can now upload to GitHub.
git add .
git commit -m "AmbedkarGPT final assignment"
git push




## DEVELOPER - Vishnupriya Murki 
