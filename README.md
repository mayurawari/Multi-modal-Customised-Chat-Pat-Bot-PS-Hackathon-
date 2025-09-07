# 🤖 Multimodal Chatbot – Two-Pane AI Workspace

A **next-generation multimodal chatbot** built with **Streamlit**, combining **text, voice, images, and AI-generated visuals** into one sleek workspace.

- 💬 **Smart AI Chat** (Gemini, OpenAI GPT-4o, Groq LLaMA)
- 🖼️ **Image Upload + Vision Q&A**
- 🎤 **Voice-to-Text with Whisper / Gemini STT**
- 🎨 **Text-to-Image Studio** powered by Gemini Imagen
- 🌐 **Web Search Router (Tavily)** for up-to-date answers
- 📂 **Persistent User Memory** (stores name & preferences locally)

✨ Designed with a **two-pane layout**:

- **Left Pane:** Chat, media upload, and voice inputs
- **Right Pane:** Dedicated **Text-to-Image Studio**

---

## 🚀 Live Demo

🔗 **App Deployment:** [Click here to try](https://multimodelchat-pat-bot.streamlit.app/)  
📹 **Explanatory Video:** [Watch Demo](https://drive.google.com/file/d/1p14eHT047kCprXtX24xGdC0PmRH7bg9k/view?usp=sharing)  

---

## 📖 Features

✅ **Multi-Model Support**

- Google **Gemini 2.5 Flash**
- OpenAI **GPT-4o Mini**
- Groq **LLaMA-3.3-70B Versatile**

✅ **Multimodal Capabilities**

- Upload an **image** and ask questions about it (Vision AI)
- Record **voice messages** and auto-transcribe with STT
- Generate **AI images** with styles and aspect ratios

✅ **Smart Routing**

- Built-in **prompt router** for concise, polished responses
- **Web search integration** with Tavily for fresh answers

✅ **Privacy-first API Handling**

- API keys loaded from `st.secrets`
- Optional in-session overrides (masked, never saved permanently)

✅ **Modern UI**

- Two-pane **chat + creative studio** layout
- Persistent **user name memory**
- Minimalist loaders, professional design, and responsive cards

---

## 🛠️ Tech Stack

- **Frontend / App:** [Streamlit](https://streamlit.io/)
- **AI Models:**
  - Google [Gemini](https://ai.google.dev/)
  - [OpenAI](https://openai.com/) GPT Models
  - [Groq](https://groq.com/) LLaMA
- **Web Search:** [Tavily API](https://tavily.com/)
- **Utilities:**
  - `PIL` for images
  - `dotenv` for local development
  - Persistent JSON storage for memory

---

## ⚙️ Installation

Clone the repo:

```bash
git clone https://github.com/your-username/multimodal-chatbot.git
cd multimodal-chatbot
Create a virtual environment & install dependencies:

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

Set up your API keys in .streamlit/secrets.toml:

OPENAI_API_KEY = "your-openai-key"
GEMINI_API_KEY = "your-gemini-key"
GROQ_API_KEY = "your-groq-key"
TAVILY_API_KEY = "your-tavily-key"


Run the app:

streamlit run app.py

🎯 Usage

Start chatting with your chosen AI model

Upload an image → ask questions about it

Record your voice → get instant transcript & response

Open the Text-to-Image Studio → generate stunning visuals

📂 Project Structure
📦 multimodal-chatbot
 ┣ 📜 app.py              # Main Streamlit app
 ┣ 📜 requirements.txt    # Python dependencies
 ┣ 📜 README.md           # Project docs
 ┗ 📂 .streamlit
    ┗ 📜 secrets.toml     # API keys (local only)

🔑 API Keys

This app uses multiple providers:

Gemini (for text, vision, STT, and TTI)

OpenAI (chat + Whisper STT)

Groq (chat completions)

Tavily (web search)

👉 Keys are loaded from st.secrets. You may override them in session only.

📹 Demo Video

🎥 Watch Full Walkthrough

📸 Screenshots

🖼️ Chat Interface:


🎨 Text-to-Image Studio:


🤝 Contributing

Contributions are welcome! 🚀

Fork the repo

Create a feature branch (git checkout -b feature-name)

Commit changes (git commit -m 'Add new feature')

Push to branch (git push origin feature-name)

Create a Pull Request

⭐ Support

If you find this project helpful:

🌟 Star this repo

🔗 Share it with your network

🐞 Open issues for bugs / suggestions
```


