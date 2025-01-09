import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import speech_recognition as sr
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


def color_generator(index: int, max_index: int):
    color = int(index * int(np.floor(255 / max_index)))
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
            cell_value = numbers_matrix[i][j]
            text = str(cell_value)
            grid = True
            if not text.isdigit():
                grid = False
                color = colors_dict.get(cell_value, (255, 255, 255))
            else:
                color = grayscale_colors_dict[cell_value]
                color = grayscale_to_rgb(color)
            x0, y0 = j * cell_size, i * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="black" if grid else None)

            # Add numbers to each block
            if not text.isdigit():
                text = ""
            bbox = draw.textbbox((0, 0), text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x = x0 + (cell_size - text_width) // 2
            text_y = y0 + (cell_size - text_height) // 2
            draw.text((text_x, text_y), text, fill="black",font=ImageFont.truetype("arial.ttf", 28))

    return img
st.set_page_config(
    page_title="ملاذّ|Malath",  
    page_icon="🎨", 
)

# Apply RTL styling for the entire page and selectbox
st.markdown(
    """
    <style>
        body {
            direction: rtl;
            text-align: right;
        }
        .stSelectbox label {
            direction: rtl;
            text-align: right;
        }
        .stSelectbox div {
            direction: rtl;
        }
        .stSelectbox select {
            text-align: right;
        }
        .stMarkdown {
            direction: rtl;
        }
    </style>
""",
    unsafe_allow_html=True,
)

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
    st.image("images\Malath logo.png", width=670)  # Add the logo image (left column)
    st.markdown("""
        ##### عن ملاذ
        "ملاذ" هو منصة تفاعلية حيث يمكنك إحياء أفكارك الإبداعية عن طريق تلوين مناطق مختلفة في شبكة باستخدام أوامر صوتية.
         مع نهجنا، يمكنك التحدث  لاختيار المنطقة واللون، ومتابعة تحديث رسمتك فورًا!

        ##### 👩‍🎨كيفية استخدام الموقع:
        1. اختر صورة لتوينها من القائمة المنسدلة.
        2. اضغط على زر "تسجيل" لبدء الأمر الصوتي.
        3. تحدث بوضوح عن رقم المنطقة واللون الذي تريد استخدامه (بالعربية).
        4. شاهد تحديث رسمتك بالألوان الجديدة في الوقت الفعلي!
        _______________________________________________________________________________
        
         </div>
        """, unsafe_allow_html=True)




    selected_image = st.selectbox("اختر صورة التي تريد تلوينها يا بطل 👏:", pixel_images.keys())
####
    # Generate initial grid
    if ( 
        "selected_image" not in st.session_state
        or st.session_state.selected_image != selected_image
    ):
        st.session_state.selected_image = selected_image
        st.session_state.color_map = pixel_images[st.session_state.selected_image][
            "available_colors"
        ]
        numbers_matrix = pixel_images[st.session_state.selected_image]["pixel_matrix"]
        st.session_state.pixel_matrix = [row[:] for row in numbers_matrix]
        st.session_state.pixel_matrix_colored = [row[:] for row in numbers_matrix]
        st.session_state.grayscale_colors_dict = {
            i: color_generator(i, np.array(numbers_matrix).max())
            for i in np.unique(numbers_matrix)
        }

    # Display the grid image
    col1, col2 = st.columns(2)

    # Display the grid image on the left column
    with col1:
        # st.write(st.session_state.pixel_matrix_colored)
        img = st.empty()
        img.image(
            image_grid_generator(
                st.session_state.pixel_matrix_colored,
                st.session_state.color_map,
                st.session_state.grayscale_colors_dict,
            ),
            caption="الرسمة المختاره🔥",
            use_container_width=True,
        )

    # Display the color palette on the right column
    with col2:
        palette_path = pixel_images[selected_image]["color_platte"]["link"]
        st.image(
            palette_path,
           
            use_container_width=True,
        )
    st.markdown("""
         <h3 style="
        font-size: 36px; 
        font-weight: bold; 
        background: linear-gradient(90deg, red, orange, green, blue, indigo, violet); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        display: inline-block;
        margin: 0;"> بصوتك، لوّن عالمك 🎨</h3>  <!-- Dark blue color with an emoji -->
            
        
         </div>
        """, unsafe_allow_html=True)

    command = ""
    listened = False
    if st.button("اضغط للتسجيل"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("🎤 نستمع اليك قُل رقم المنطقة واللون ...")
            listened = True
            try:
                audio = recognizer.listen(source, timeout=10)
                command = recognizer.recognize_google(audio, language="ar")
                st.success(f"✅👌 تم تسجيل ابداعك: {command}")
            except Exception as e:
                st.error(f"❌ لم يتم التعرف على الصوت: {str(e)}")
        Detect=st.empty()
        detected_color = None
        for color in list(st.session_state.color_map.keys()):
            if color in command:
                detected_color = color
                break

        if detected_color:
            st.write(f"🎨 اللون المكتشف: {detected_color}")

            # Detect zone (Arabic number) using if statements
            detected_zone = None
            for word in command.split():
                if word in arabic_number_map:
                    detected_zone = arabic_number_map[word]
                    break
            if detected_zone is not None:
                st.write(f"📍 المنطقة المكتشفة: {detected_zone}")

                # Update grid
                updated_matrix = [row[:] for row in st.session_state.pixel_matrix]
                for y in range(len(updated_matrix)):
                    for x in range(len(updated_matrix[y])):
                        if updated_matrix[y][x] == detected_zone:
                            updated_matrix[y][x] = detected_color
                        else:
                            updated_matrix[y][x] = st.session_state.pixel_matrix_colored[
                                y
                            ][x]

                st.session_state.pixel_matrix_colored = updated_matrix
                img.image(
                    image_grid_generator(
                        st.session_state.pixel_matrix_colored,
                        st.session_state.color_map,
                        st.session_state.grayscale_colors_dict,
                    ),
                    caption="📷 تم تحديث الشبكة",
                    use_container_width=True,
                )
            else:
                st.error("❌ لم يتم التعرف على رقم المنطقة بشكل صحيح.")
        else:
            st.error("❌ اللون المكتشف غير موجود في القائمة.")
    elif listened:
        st.error("❌ لم يتم تسجيل أمر صوتي.")


if __name__ == "__main__":
    main()
