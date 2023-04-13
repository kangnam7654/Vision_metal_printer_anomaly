from pathlib import Path
import os

class GetProjectPath:
    def __init__(self):
        self.root_dir = self.get_project_root()
        self.data_folder = self.get_data_folder()
        self.pickle_folder = self.get_pickle_folder()
        self.current_directory = self.get_current_directory()

    @staticmethod
    def get_current_directory():
        return os.path.curdir

    @staticmethod
    def get_project_root(*paths):
        project_root = os.path.join(Path(__file__).parents[2], *paths)
        return project_root

    def get_pickle_folder(self, *paths):
        pickle_folder = self.get_project_root('pickle', *paths)
        return pickle_folder

    def get_data_folder(self, *paths):
        data_folder = os.path.join(self.root_dir, 'data', *paths)
        return data_folder

if __name__ == '__main__':
    pass
