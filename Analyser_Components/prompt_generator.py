import google.generativeai as gai
from PIL import Image
import supervision as sv
from tqdm.notebook import tqdm
import pandas as pd


class prompt_generator():
        def __init__(self,model,key,images_dir = "./data/",images_extensions = ['jpg', 'jpeg', 'png']):
            # self.model = model
            self.model = gai.GenerativeModel(model)
            self.key = key
            self.images_dir = images_dir
            self.images_extensions = images_extensions

            gai.configure(api_key=self.key)
            self.image_paths = sv.list_files_with_extensions(
            directory=self.images_dir,
            extensions=self.images_extensions)
            
            self.prompts_dict = {}

        def generate_prompts(self,number_of_prompts=20):
            for image_path in tqdm(self.image_paths):
                image_name = image_path.stem  # 'stem' gives the file name without extension
                image = Image.open(image_path)
                self.prompts_dict[image_name] = []
                for i in range(number_of_prompts):
                    Q="segmentation prompt"
                    res=self.model.generate_content([Q,image])
                    self.prompts_dict[image_name].append(res.text)

            return self.prompts_dict