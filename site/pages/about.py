import streamlit as st
from pathlib import Path


def read_md(md_file):
	return Path(md_file).read_text(encoding="utf-8")

def app():
	md_readme = read_md("./../README.md")
	st.markdown(md_readme, unsafe_allow_html=True)
