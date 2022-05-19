import pandas as pd
from absl import app

from data.load_data import compas

def main(argv):

    data = compas()
    print(pd.get_dummies(data))

if __name__ == '__main__':
    app.run(main)
