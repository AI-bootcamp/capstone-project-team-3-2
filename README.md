# 🎨 Malath - Voice-Controlled Coloring Platform

### 🧩 **Overview**
**Malath** is an interactive platform that enables users to color various grid areas using **voice commands** in Arabic. By leveraging advanced voice recognition, real-time updates, and interactive UI elements, Malath transforms creative ideas into dynamic pixel art.

### ✨ **Features**
- 🎤 **Voice Recognition**: Convert Arabic speech commands into actionable color instructions using `SpeechRecognition`.
- 🌈 **Real-Time Grid Coloring**: Watch the grid dynamically update as you issue commands.
- 🖼️ **Grid Image Selection**: Choose from a variety of predefined pixel art grids.
- 🖌️ **Color Palette Integration**: Customize your artwork with diverse color palettes.
- 🔤 **RTL and Arabic Language Support**: Fully compatible with right-to-left text and Arabic input.

---

### 📁 **File Structure**

```
├── images/                        # Color palette images
│   ├── Bear-ColorPlatte.png
│   ├── butterfly-ColorPlatte.png
│   ├── Dolphin-ColorPlatte.png
│   ├── fox-ColorPlatte.png
│   ├── MARIO-ColorPlatte.png
│   ├── Sonic-ColorPlatte.png
│   ├── spiderMan-ColorPlatte.png
│   └── watermelon-ColorPlatte.png
├── venv/                          # Virtual environment folder
├── appGrid+ColorPlatte.py         # Streamlit app for managing grid and colors
├── images.py                      # Utility functions for handling images
├── images+ColorPlatte.py          # Extended functionality for grids and palettes
├── Logo.png                       # Project logo
├── README.md                      # Project documentation
└── requirements.txt               # Python package dependencies
```

---

### 🎯 **Core Functionalities**
1. **Voice Recognition**:
   - Process Arabic voice commands using Google Speech Recognition to identify grid zones and colors.

2. **Grid Coloring**:
   - Dynamically generate and color grids using Pillow.

3. **Interactive UI**:
   - Streamlit-based interface allows seamless image selection and grid updates.

4. **Color Mapping**:
   - Map Arabic color names to RGB values and apply them to grid zones.

5. **RTL and Arabic Language Support**:
   - Ensure proper text alignment and functionality for Arabic-speaking users.

---

### ⚙️ **Setup and Usage**

#### 🚰 **Prerequisites**
- **Python 3.8+**
- **Virtual Environment** (recommended)
- Required Libraries: `streamlit`, `numpy`, `Pillow`, `speechrecognition`

#### 💻 **Installation**

   1. Clone the repository:
      ```bash
      git clone https://github.com/AI-bootcamp/capstone-project-team-3-2.git
      cd capstone-project-team-3-2
      ```

Your repository is now cloned locally, and you're inside the project folder. Let me know if you need further assistance with setting up the repository or working on the project!

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run appGrid+ColorPlatte.py
   ```

---

### 🧑‍💻 **Usage**

1. Launch the Streamlit application.
2. Select a grid image from the dropdown menu.
3. Press the **Record** button and speak your command in Arabic (e.g., "أربعة لون أخضر").
4. The grid will dynamically update with the specified color and zone.

---

### 📊 **Example Commands**
- **Command**: "رقم ثلاثة لون أحمر"
  - *Effect*: Colors zone 3 in red.
- **Command**: "رقم واحد لون أزرق"
  - *Effect*: Colors zone 1 in blue.

---

### 🚀 **Future Enhancements**
- 🔬 **AI-Powered Color Suggestions**: Integrate NLP models to suggest colors based on user context.
- 🌍 **Multi-Language Support**: Expand functionality to accommodate additional languages.
- 🎨 **Custom Grid Uploads**: Allow users to upload and color their own images.
- 📊 **Usage Insights**: Provide analytics for frequently used commands and colors.

---

### 🤝 **Contributing**
Contributions are welcome! Submit pull requests or report issues to enhance the project.

