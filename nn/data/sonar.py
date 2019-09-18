import pandas as pd

class SonarData(object):

    def __init__(self):
        pass

    def load_data(self, file="D:\\Projects\\PythonProjects\\DeepLearning\\data\sonar.all-data"):
        df = pd.read_csv(file, header=None)
        df.loc[df[60] == 'M', 60] = 1
        df.loc[df[60] == 'R', 60] = -1
        self.__sonar_df = df

    def get_output_vector(self):
        return self.__sonar_df[60].to_numpy()

    def get_input_data(self, *columns):
        for column in columns:
            if column < 0 or column > 59:
                raise AttributeError(f"{column} should be between 0 to 59")
        return self.__sonar_df[list(columns)].to_numpy()

    def classified_object(self, value):
        if value > 0:
            return "Mine"
        else:
            return "Rock"

