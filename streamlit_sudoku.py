import streamlit as st
from selenium import webdriver
from PIL import Image
import sys
sys.path.append("../")
from src.func import *


image = Image.open("Images/Cabecera.jpg")
st.image(image)
st.markdown("<h1 style='color: green;'>SUPER SUDOKU SOLVER</h1>", unsafe_allow_html=True)

st.write("""
## Upload an image with a sudoku and find the solution!
""")

st.write("""
### Are you trying to solve a very difficult sudoku but you are stuck? You could give up...
DON'T DO IT !!!
""")

st.write("""
This is a sudoku solver, if you upload an image in .jpg or .png format,
it will return the solution. Please, do not try to upload photos, as the scanner 
is still under construction.
""")

st.write("## Upload an image: ")
sudoku = st.file_uploader("Select file", type = ['jpg', 'png'])
if sudoku:
    st.write('Sudoku uploaded')
    img = Image.open(sudoku)
    st.image(img)



if st.button("Solve it!"):
    img_path = "Sudoku_web.png"
    img = img.save(img_path)
    try:
        cut = sudoku_cut_frame(img_path)
        list_pics = sudoku_split_81(cut)
        final_list, probability_numbers, number_pics = from_pic_to_numbers(list_pics)
        result = solver(final_list)
        l = []
        for i in result:
            num = i[1]
            l.append(num)  
        st.write("## The solved sudoku:")    
        st.write(
                f"""- - - - - - - - - - -
                {l[0]} {l[1]} {l[2]} | {l[3]} {l[4]} {l[5]} | {l[6]} {l[7]} {l[8]}
                {l[9]} {l[10]} {l[11]} | {l[12]} {l[13]} {l[14]} | {l[15]} {l[16]} {l[17]}
                {l[18]} {l[19]} {l[20]} | {l[21]} {l[22]} {l[23]} | {l[24]} {l[25]} {l[26]}
                - - - - - - - - - - -
                {l[27]} {l[28]} {l[29]} | {l[30]} {l[31]} {l[32]} | {l[33]} {l[34]} {l[35]}
                {l[36]} {l[37]} {l[38]} | {l[39]} {l[40]} {l[41]} | {l[42]} {l[43]} {l[44]}
                {l[45]} {l[46]} {l[47]} | {l[48]} {l[49]} {l[50]} | {l[51]} {l[52]} {l[53]} 
                - - - - - - - - - - -
                {l[54]} {l[55]} {l[56]} | {l[57]} {l[58]} {l[59]} | {l[60]} {l[61]} {l[62]}
                {l[63]} {l[64]} {l[65]} | {l[66]} {l[67]} {l[68]} | {l[69]} {l[70]} {l[71]}
                {l[72]} {l[73]} {l[74]} | {l[75]} {l[76]} {l[77]} | {l[78]} {l[79]} {l[80]}"""
                )
        st.write("Thanks for using Super Sudoku Solver!") 
    except Exception:
        st.write("Ups, something happened! :(")