import time
import streamlit as st

from core.text_loader import read_random_text_file
from core.gpt_generator import GPT3Generator
from core.prompts import DEFAULT_PROMPT


st.set_page_config(page_title="Generate Rasa NLU Demo", layout="wide")


def main():
    user_input = ""
    gpt3_generator = GPT3Generator()

    p1, _ = st.columns([1, 1])
    prompt_area = p1.empty()
    sys_prompt = prompt_area.text_area("Prompt", value=DEFAULT_PROMPT.template)

    head1, head = st.columns([1, 1])

    with head1:
        st.header("Enter FAQ article:")
        text_area = st.empty()
        if st.session_state.get("content"):
            user_input = text_area.text_area(
                " " * 6, st.session_state.content, height=400
            )
        else:
            user_input = text_area.text_area("" * 7, value="", height=400)

    with head:
        st.header("Result:")
        result_area = st.empty()

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 2])

    if col1.button("Generate from Random Article"):
        content = read_random_text_file()
        text_area.text_area(" ", content, height=400)
        st.session_state["content"] = content

    if col3.button("Generate NLU !"):
        start_time = time.time()
        if user_input:
            is_parsing_success, generated_nlu = gpt3_generator.generate(
                user_input, sys_prompt
            )
        elif st.session_state.get("content"):
            is_parsing_success, generated_nlu = gpt3_generator.generate(
                st.session_state["content"], sys_prompt
            )

        result_area.text_area("   ", value=generated_nlu, disabled=True, height=400)

        col5.write(f"Cost time: {time.time() - start_time:.2f} seconds")

        if is_parsing_success:
            download_btn_label = "Download Result"
        else:
            download_btn_label = "⚠️(Parsing Failed)Download Result"

        with st.container():
            st.write("     ")
            col5.download_button(
                label=download_btn_label,
                data=generated_nlu,
                file_name="result.yaml",
                mime="text/yaml",
            )


if __name__ == "__main__":
    main()
