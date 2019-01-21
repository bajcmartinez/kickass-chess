# Kickass chess
A weekend project to make a computer beat me in a chess game

# Set up locally
1. create require directories
    ```
    mkdir ai/data
    mkdir ai/datasets
    ```

2. Download the games that will be used to train the network
    ```
    I download the pgn files from: http://kingbase-chess.net/
    Copy all the files into ai/data
    ```

3. Generate the datasets, the network needs to serialize the games in order to train the network, this is what we will be doing on the next step
    ```
    cd ai
    python generate_datasets.py
    
    (this may take up a few minutes..., if you want to tune the number of samples please review code)
    ```
    
4. All is ready to train the network
    ```
    In th ai folder run:
    python train.py
    
    (this will take a while depending on your computer configuration, I recommend you to use a GPU)
    ```
    
 5. Run the game :)
    ```
    On the project root:
    
    FLASK_APP=game.py python flask run
    ```
    
 6. Game time!