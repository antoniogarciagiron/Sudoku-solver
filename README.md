# Sudoku-solver

![Head](https://github.com/antoniogarciagiron/Sudoku-solver/blob/main/Images/Cabecera.jpg)


## Introduction:

This is my final project done during the [IronHack](https://www.ironhack.com/en) Data Analytics bootcamp.  
The objective was to create an algorithm to take images with a sudoku and return the solution.  
The rules to solve sudokus are simple, there are 9 rows, 9 columns and 9 sub-frames, and each one can have 9 numbers, but all may be different, from 1 to 9.  

Why did I decide to make a sudoku solver? There are two main reasons. First, my father likes sudokus and I wanted to make this for him. The second reason, we had 10 days to make a project from the scratch, this was the idea I most liked, it was a real challenge, and I like challenges.

![pic1](https://github.com/antoniogarciagiron/Sudoku-solver/blob/main/Images/Diapositiva2.JPG)


## Data: 

The following Python libraries were used:
- [sklearn](https://scikit-learn.org/stable/)
- [pandas](https://pandas.pydata.org/docs/user_guide/index.html)
- [seaborn](https://seaborn.pydata.org/index.html)
- [matplotlib.pyplot](https://matplotlib.org/stable/contents.html)
- [selenium](https://www.selenium.dev/documentation/en/)
- [PIL](https://pillow.readthedocs.io/en/stable/)
- [cv2](https://docs.opencv.org/master/)
- [imutils](https://pypi.org/project/imutils/)


## Image manipulation: 

The main challenge was to transform an image with a sudoku into a list of numbers. 
The process is shown in the next image:

![pic2](https://github.com/antoniogarciagiron/Sudoku-solver/blob/main/Images/Diapositiva3.JPG)  

First, the external frame must be found. Then, the image is divided into 81 squares, one for each number/empty space in the sudoku. Once we have the original sudoku divided into small images we need to check if there is a number or not, for this, we calculate the average pixel colour. Then we create a list, if there is an empty space in the sudoku it will add a 0, otherwise it will add the number.  
But how do we know what number we have in the image? For this we need to train a neuronal network. In a first approach I tried to use the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) database, however, as the number in MNIST are handwritten, there were lots of errors, in particular confusing 1s with 7s and 2s with 5s. For this reason I decided to create my own dataset, using different text fonts and creating random numbers from 1 to 9, changing the size, position and angle. Once I had 40000 numbers, the neuronal network was trained and used to guess the numbers in the pictures.

![pic3](https://github.com/antoniogarciagiron/Sudoku-solver/blob/main/Images/Diapositiva4.JPG)


## Sudoku solver: 

With the list of numbers and 0s for each empty position, I coded an algorithm to solve the sudoku. Basically, for each 0 it finds the possible numbers depending on the row, column and quadrant where the 0 is placed, it gives the first option and goes to the next 0, gives the first possible value... and so on. However, as we are giving the first value each time, it may be or not the correct number, and it will reach a 0 where there will be no options... but we know that there is always a solution. Thus, the absence of options means that we did a mistake in a previous step, so the algorithm comes back to the previous 0, selects the second option, and continues. If there is no option, it comes back again, and by repeating this iterations it finds the correct solution in a few seconds.

![pic4](https://github.com/antoniogarciagiron/Sudoku-solver/blob/main/Images/Diapositiva5.JPG)
![pic5](https://github.com/antoniogarciagiron/Sudoku-solver/blob/main/Images/Diapositiva6.JPG)

A streamlit file was prepared and ready to be uploaded. It consist in a website where the user can upload a picture with a sudoku and receive the solution.


## Future work:

- Scanner: to solve sudokus directly with your phone camera! The target is to find the sudoku frame, and reshape it to a square
- Improve the algorithm: make it faster
- Return the image with numbers in the empty places when you get the solution, instead of the numbers in the screen
- Defensive programming: what if I add a picture without sudoku? Or a sudoku without solution… ? This stuff has to be considered in the future
