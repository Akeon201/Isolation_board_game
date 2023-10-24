# Implementing Adversarial Search in the board game, *Isolation*
#### Author: Kenyon Leblanc
This README will guide you through setting up and using the program.

## Setup

    1. You must have Python3 installed.
    2. Tkinter should come pre-packaged with python. If not, try "sudo apt-get install python3-tk" on linux.
    3. Ensure you are in the same file directory as the main.py file.

## Run Game

    1. Run the command "python3 ./main.py" to run the program.

## How to Play
    
    1. At the bottom of the game window will be text explaining whos turn it is.
    2. Click a tile an adjacent tile around your pawn(blue). The tile must be white, signifying a token exists there.
    3. Next select any white tile to remove a token, turning the tile black.
    4. The player with no moves remaining loses.

## Notes

    Lines 9 through 33 in the main.py file are reserved for togglable settings such as AI vs AI, changing heuristics 
    of AI, displaying advanced statistics in console.
