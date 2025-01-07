import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import speech_recognition as sr
from transformers import pipeline
from images import pixel_images


def grayscale_to_rgb(grayscale: int) -> tuple:
    """
    Convert a grayscale value (0-255) to an RGB tuple.

    Parameters:
        grayscale (int): A number between 0 and 255 representing grayscale.

    Returns:
        tuple: (R, G, B) where each is equal to the grayscale value.
    """
    if 0 <= grayscale <= 255:
        return (grayscale, grayscale, grayscale)
    else:
        raise ValueError("Grayscale value must be between 0 and 255")

def color_generator(index : int, max_index: int):
    color = int(index * int(np.floor(255/max_index)))
    if index == 0:
        color = 255
    return color

# Grid Generation Function
def image_grid_generator(numbers_matrix, colors_dict, grayscale_colors_dict, cell_size=50):
    rows, cols = len(numbers_matrix), len(numbers_matrix[0])
    img_width = cols * cell_size
    img_height = rows * cell_size

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    for i in range(rows):
        for j in range(cols):
            cell_value = numbers_matrix[i][j] # 0, 1, 2, 3
            text = str(cell_value)
            grid = True
            if not text.isdigit():
                grid = False
                color = colors_dict.get(cell_value, (255, 255, 255))  # Default to white
            else:
                color = grayscale_colors_dict[cell_value]
                color = grayscale_to_rgb(color)
            x0, y0 = j * cell_size, i * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="black" if grid else None)

            # Add numbers to each block
            if not text.isdigit():
                text=''
            bbox = draw.textbbox((0, 0), text)  # Get text bounding box
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x = x0 + (cell_size - text_width) // 2
            text_y = y0 + (cell_size - text_height) // 2
            draw.text((text_x, text_y), text, fill="black")

    return img

# Arabic number map (0 to 5)
arabic_number_map = {
    "صفر": 0,
    "واحد": 1,
    "اثنين": 2,
    "ثلاثه": 3,
    "اربعه": 4,
    "خمسه": 5,
}

# Streamlit UI
def main():
    st.title("Interactive Grid Coloring with Speech Recognition")
    st.markdown("Speak a region number (in Arabic) and a color to update the grid.")
    nlp_model = pipeline("zero-shot-classification")

    selected_image = st.selectbox(
        'Select an image:',
        pixel_images.keys()
    )


    # Generate initial grid
    if("selected_image" not in st.session_state or st.session_state.selected_image != selected_image):
        st.session_state.selected_image = selected_image
        st.session_state.color_map = pixel_images[st.session_state.selected_image]['available_colors']
        numbers_matrix = pixel_images[st.session_state.selected_image]['pixel_matrix']
        st.session_state.pixel_matrix = [row[:] for row in numbers_matrix]
        st.session_state.pixel_matrix_colored = [row[:] for row in numbers_matrix]
        st.session_state.grayscale_colors_dict = {i: color_generator(i, np.array(numbers_matrix).max()) for i in np.unique(numbers_matrix)}

    st.image(
        image_grid_generator(st.session_state.pixel_matrix, st.session_state.color_map, st.session_state.grayscale_colors_dict),
        caption="Initial Grid",
        use_column_width=True,
    )

    st.header("🎤 Record Voice Commands")
    st.write("Available colors: ", ", ".join(st.session_state.color_map.keys()))
    command = ""
    listened = False
    if st.button("Press to Record"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("🎤 Listening...")
            listened = True
            try:
                audio = recognizer.listen(source, timeout=10)
                command = recognizer.recognize_google(audio, language="ar")
                st.success(f"✅ Recorded Command: {command}")
            except Exception as e:
                st.error(f"❌ Could not recognize the audio: {str(e)}")

    if command:
        st.write(f"🔍 Processing Command: {command}")
        # Detect color from command
        color_candidates = list(st.session_state.color_map.keys())
        nlp_result = nlp_model(command, candidate_labels=color_candidates)
        detected_color = nlp_result["labels"][0]  # Most likely color

        if detected_color in st.session_state.color_map:
            st.write(f"🎨 Detected Color: {detected_color}")

            # Detect zone (Arabic number) from command
            detected_zone = None
            for word in command.split():
                if word in arabic_number_map:
                    detected_zone = arabic_number_map[word]
                    break

            if detected_zone is not None:
                st.write(f"📍 Detected Zone: {detected_zone}")

                # Update grid
                updated_matrix = [row[:] for row in st.session_state.pixel_matrix]
                for y in range(len(updated_matrix)):
                    for x in range(len(updated_matrix[y])):
                        if updated_matrix[y][x] == detected_zone:
                            updated_matrix[y][x] = detected_color
                        else:
                            updated_matrix[y][x] = st.session_state.pixel_matrix_colored[y][x]

                st.session_state.pixel_matrix_colored = updated_matrix

                st.image(
                    image_grid_generator(st.session_state.pixel_matrix_colored, st.session_state.color_map, st.session_state.grayscale_colors_dict),
                    caption="📷 Updated Grid",
                    use_column_width=True,
                )
            else:
                st.error("❌ Could not detect a valid zone number.")
        else:
            st.error("❌ Detected color is not in the recognized list.")
    elif listened:
        st.error("❌ No voice command recorded.")

# Run the app
if __name__ == "__main__":
    main()